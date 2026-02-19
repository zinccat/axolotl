"""
Expert Parallel (EP) + ScatterMoE forward for Qwen3-MoE (and compatible models).

Uses ScatterMoE's grouped GEMM kernels (parallel_linear / flatten_sort_count) for
efficient local expert computation, wrapped with all_to_all token dispatch/combine
for cross-rank Expert Parallelism.

Compatible interface (attributes read from the MoE block):
  self.gate.weight           [num_experts, hidden_dim]
  self.gate.top_k            int
  self.gate.num_experts      int
  self.gate.norm_topk_prob   bool
  self.experts.gate_up_proj  [num_experts, 2*intermediate, hidden]
  self.experts.down_proj     [num_experts, hidden, intermediate]
  self.experts.act_fn        callable
  self.shared_expert         optional (Qwen2-MoE style)
  self.shared_expert_gate    optional
"""
from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed._functional_collectives import all_to_all_single_autograd
from torch.distributed.device_mesh import init_device_mesh

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


# ---------------------------------------------------------------------------
# Lazy accessor for ScatterMoE ops
# ---------------------------------------------------------------------------

def _get_scattermoe_ops():
    """Lazy-load parallel_linear and flatten_sort_count from the loaded kernel module.

    The kernel module name is 'scattermoe_{path_hash}' in sys.modules after the
    kernels library loads it.  We find it by looking for the required attributes
    and verifying they are callable (to skip PyTorch internal _OpNamespace objects).
    """
    for mod in sys.modules.values():
        if hasattr(mod, "parallel_linear") and hasattr(mod, "flatten_sort_count"):
            pl = mod.parallel_linear
            fsc = mod.flatten_sort_count
            if callable(pl) and callable(fsc):
                return pl, fsc
    raise ImportError(
        "ScatterMoE kernel not loaded. "
        "Ensure use_scattermoe=True in the plugin config so that "
        "_register_kernels() is called before apply_ep_scattermoe()."
    )


# ---------------------------------------------------------------------------
# Utility: look up the MoE block class for a given HuggingFace model type
# ---------------------------------------------------------------------------

def get_model_moe_block(model_type: str):
    """Return the SparseMoeBlock class for the given transformers model type.

    Args:
        model_type: e.g. ``"qwen3_moe"``, ``"qwen2_moe"``, ``"mixtral"``.
    """
    from axolotl.utils.callbacks.models import get_causal_lm_model_cls_prefix

    module_path = f"transformers.models.{model_type}.modeling_{model_type}"
    model_cls_prefix, _ = get_causal_lm_model_cls_prefix(model_type)
    module = __import__(module_path, fromlist=[f"{model_cls_prefix}SparseMoeBlock"])
    return getattr(module, f"{model_cls_prefix}SparseMoeBlock")


# ---------------------------------------------------------------------------
# EP + ScatterMoE forward
# ---------------------------------------------------------------------------

class _EPScatterMoeForward(nn.Module):
    """Stateless class whose forward() is bound to Qwen3MoeSparseMoeBlock instances.

    Do NOT instantiate directly.  Use apply_ep_scattermoe() which calls
    types.MethodType(forward, module) on each MoE block.

    The target module must carry a ``_ep_mesh`` attribute (a 1-D DeviceMesh
    spanning the EP process group) set by apply_ep_scattermoe().
    """

    @staticmethod
    def forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:  # noqa: N805
        parallel_linear, flatten_sort_count = _get_scattermoe_ops()

        ep_mesh: DeviceMesh = self._ep_mesh  # type: ignore[attr-defined]
        ep_group = ep_mesh.get_group()
        ep_rank = dist.get_rank(ep_group)
        ep_size = dist.get_world_size(ep_group)

        B, S, H = hidden_states.shape
        N = B * S
        hidden = hidden_states.view(N, H)

        # ── Shared expert (Qwen2-MoE style, absent in Qwen3-MoE) ──────────────
        shared_out: torch.Tensor | None = None
        if getattr(self, "shared_expert", None) is not None:
            shared_out = F.sigmoid(self.shared_expert_gate(hidden)) * self.shared_expert(hidden)

        # ── 1. Routing (replicated on all EP ranks) ───────────────────────────
        router_logits = F.linear(hidden, self.gate.weight)          # [N, E]
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        top_k: int = self.gate.top_k
        num_experts: int = self.gate.num_experts
        num_local_experts: int = num_experts // ep_size

        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        if self.gate.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(-1, keepdim=True)
        routing_weights = routing_weights.to(hidden.dtype)          # [N, top_k]

        # ── 2. Flatten & sort by global expert for contiguous all_to_all ──────
        #
        # Each of the N tokens is duplicated top_k times, once per expert slot.
        #   flat_tokens[i]   = the token assigned to flat_expert_ids[i]
        #   flat_weights[i]  = routing weight for that (token, expert) pair
        flat_tokens  = hidden.repeat_interleave(top_k, dim=0)      # [N*top_k, H]
        flat_weights = routing_weights.reshape(-1)                  # [N*top_k]
        flat_expert_ids = selected_experts.reshape(-1).int()        # [N*top_k]

        with torch.no_grad():
            sort_order = flat_expert_ids.argsort(stable=True)       # [N*top_k]
            sorted_expert_ids = flat_expert_ids[sort_order]
            num_tpe = torch.bincount(sorted_expert_ids, minlength=num_experts).int()
            # input_splits[r] = #tokens this rank sends to EP rank r
            input_splits = num_tpe.view(ep_size, num_local_experts).sum(1).tolist()

        sorted_tokens  = flat_tokens[sort_order]                    # [N*top_k, H]
        sorted_weights = flat_weights[sort_order]                   # [N*top_k]

        # ── 3. EP dispatch (all_to_all) ────────────────────────────────────────
        with torch.no_grad():
            # Exchange count metadata (no gradient needed)
            recv_counts = torch.empty_like(num_tpe)
            dist.all_to_all_single(recv_counts, num_tpe, group=ep_group)
            # output_splits[r] = #tokens this rank receives from EP rank r
            output_splits = recv_counts.view(ep_size, num_local_experts).sum(1).tolist()
            T_local = sum(output_splits)

        # Dispatch token embeddings — gradients must flow through this
        recv_tokens = all_to_all_single_autograd(
            sorted_tokens,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=ep_group,
        )  # [T_local, H]

        # Dispatch routing weights (no grad needed; applied during down-proj)
        with torch.no_grad():
            recv_weights = torch.empty(
                T_local, dtype=sorted_weights.dtype, device=sorted_weights.device
            )
            dist.all_to_all_single(
                recv_weights, sorted_weights,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=ep_group,
            )  # [T_local]

        # ── 4. Reconstruct local expert ids for received tokens ────────────────
        #
        # recv_counts[s*L + e] = #tokens received from rank s for local expert e.
        # The tokens from each rank arrive sorted by local expert within that chunk.
        with torch.no_grad():
            local_recv_counts = recv_counts.view(ep_size, num_local_experts)  # [ep_size, L]
            if T_local > 0:
                local_expert_ids = torch.cat([
                    torch.full(
                        (int(local_recv_counts[s, e].item()),), e,
                        dtype=torch.int64, device=hidden.device,
                    )
                    for s in range(ep_size)
                    for e in range(num_local_experts)
                    if local_recv_counts[s, e].item() > 0
                ])  # [T_local]
            else:
                local_expert_ids = torch.empty(0, dtype=torch.int64, device=hidden.device)

            # Sort for ScatterMoE grouped-GEMM
            sorted_local_expert_idxs, sorted_local_scattered_idxs, local_expert_offsets = (
                flatten_sort_count(local_expert_ids.unsqueeze(1), num_experts=num_local_experts)
            )

        # ── 5. ScatterMoE local expert computation ─────────────────────────────
        local_start = ep_rank * num_local_experts

        # Weight slices: [num_local_experts, 2*inter, hidden] / [num_local, hidden, inter]
        # Transposed for parallel_linear: [L, hidden, 2*inter] / [L, inter, hidden]
        gate_up_t = self.experts.gate_up_proj[
            local_start : local_start + num_local_experts
        ].transpose(2, 1)

        down_t = self.experts.down_proj[
            local_start : local_start + num_local_experts
        ].transpose(2, 1)

        if T_local > 0:
            # UP + GATE projection
            # grouped_in=False: input is in receive order (not expert-sorted)
            # grouped_out=True: output is in expert-sorted order
            gates_out, h = parallel_linear(
                recv_tokens,                    # [T_local, H] — receive order
                gate_up_t,                      # [L, H, 2*inter]
                1,                              # k=1: each dispatched token → 1 expert
                sorted_local_expert_idxs,       # [T_local]
                sorted_local_scattered_idxs,    # [T_local]
                local_expert_offsets,           # [L]
                grouped_in=False,
                grouped_out=True,
            ).chunk(2, dim=-1)                  # each: [T_local, inter]

            h = self.experts.act_fn(gates_out) * h  # [T_local, inter]

            # DOWN projection, applying routing weights via gates parameter.
            # grouped_in=True: input is in expert-sorted order
            # grouped_out=False: output is in receive order (matches recv_tokens)
            # gates=[T_local, 1]: multiplies each output by its routing weight
            expert_output = parallel_linear(
                h,                              # [T_local, inter] — expert-sorted
                down_t,                         # [L, inter, H]
                1,
                sorted_local_expert_idxs,
                sorted_local_scattered_idxs,
                local_expert_offsets,
                grouped_in=True,
                grouped_out=False,
                gates=recv_weights.unsqueeze(1),  # [T_local, 1]
            )  # [T_local, H] — receive order, already weighted
        else:
            expert_output = torch.zeros(0, H, dtype=hidden.dtype, device=hidden.device)

        # ── 6. EP combine (all_to_all) — splits are reversed vs dispatch ───────
        combined = all_to_all_single_autograd(
            expert_output,
            output_split_sizes=input_splits,   # receive back what we originally sent
            input_split_sizes=output_splits,   # send back what we received
            group=ep_group,
        )  # [N*top_k, H] — in global dispatch sort order

        # ── 7. Undo global sort, reduce over top_k ─────────────────────────────
        with torch.no_grad():
            inv_sort_order = torch.empty_like(sort_order)
            inv_sort_order[sort_order] = torch.arange(
                N * top_k, device=sort_order.device
            )

        output = combined[inv_sort_order]               # [N*top_k, H] — original flat order
        output = output.view(N, top_k, H).sum(dim=1)   # [N, H] — weighted sum over top_k

        if shared_out is not None:
            output = output + shared_out

        return output.reshape(B, S, H)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_ep_scattermoe(
    model: nn.Module,
    model_type: str,
    ep_size: int,
) -> None:
    """Patch every MoE sparse-expert block in *model* with the EP+ScatterMoE forward.

    Builds a 1-D EP DeviceMesh from the current process group, stores it as
    ``_ep_mesh`` on each MoE block instance, and replaces ``forward`` with the
    EP-aware implementation.

    Args:
        model:      The loaded HuggingFace model.
        model_type: e.g. ``"qwen3_moe"``, ``"qwen2_moe"``.  Used to look up the
                    block class from ``transformers``.
        ep_size:    Expert parallel degree (must divide world_size and num_experts).
    """
    # Build the EP mesh: a single-dimension mesh of size ep_size.
    # Contiguous ranks are placed in the same EP group (favours NVLink locality).
    world_size = dist.get_world_size()
    if world_size % ep_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by expert_parallel_size ({ep_size})"
        )

    # Shape: [num_ep_groups, ep_size] — rank r is in EP group r // ep_size
    ep_mesh = init_device_mesh(
        "cuda",
        (world_size // ep_size, ep_size),
        mesh_dim_names=("dp", "ep"),
    )["ep"]  # 1-D mesh of size ep_size

    moe_block_cls = get_model_moe_block(model_type)

    patched = 0
    for module in model.modules():
        if isinstance(module, moe_block_cls):
            module._ep_mesh = ep_mesh  # type: ignore[attr-defined]
            module.forward = types.MethodType(_EPScatterMoeForward.forward, module)  # type: ignore[method-assign]
            patched += 1

    if patched == 0:
        raise RuntimeError(
            f"No {moe_block_cls.__name__} modules found in the model. "
            "Is the correct model_type set?"
        )
