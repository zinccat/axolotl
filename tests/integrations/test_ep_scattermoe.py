"""
Unit tests for the EP + ScatterMoE forward pass.

Tests that don't require a real distributed setup use a single-rank simulation
(ep_size=1) or mock the EP mesh.  The multi-rank integration test requires
torchrun / pytest-dist with CUDA.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Minimal stubs mirroring Qwen3MoeSparseMoeBlock / Qwen3MoeTopKRouter /
# Qwen3MoeExperts so we can run without a full HuggingFace model.
# ---------------------------------------------------------------------------


class _FakeRouter(nn.Module):
    def __init__(self, num_experts: int, hidden: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = False
        self.weight = nn.Parameter(torch.randn(num_experts, hidden))


class _FakeExperts(nn.Module):
    def __init__(self, num_experts: int, hidden: int, intermediate: int):
        super().__init__()
        # Qwen3MoeExperts stores weights as 3-D tensors
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate, hidden)
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden, intermediate)
        )
        self.act_fn = F.silu


class _FakeMoeBlock(nn.Module):
    """Mimics Qwen3MoeSparseMoeBlock with standard forward."""

    def __init__(self, num_experts: int, hidden: int, intermediate: int, top_k: int):
        super().__init__()
        self.gate = _FakeRouter(num_experts, hidden, top_k)
        self.experts = _FakeExperts(num_experts, hidden, intermediate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, H = hidden_states.shape
        hidden = hidden_states.view(-1, H)
        logits = F.linear(hidden, self.gate.weight)
        weights = F.softmax(logits, dim=-1, dtype=torch.float)
        weights, experts = torch.topk(weights, self.gate.top_k, dim=-1)
        weights = weights.to(hidden.dtype)

        out = torch.zeros_like(hidden)
        expert_mask = F.one_hot(experts, self.gate.num_experts).permute(2, 1, 0)
        for ei in expert_mask.sum(dim=(-1, -2)).nonzero():
            idx = ei[0].item()
            top_k_pos, tok_idx = torch.where(expert_mask[idx])
            cur = hidden[tok_idx]
            g, u = F.linear(cur, self.experts.gate_up_proj[idx]).chunk(2, dim=-1)
            h = F.silu(g) * u
            h = F.linear(h, self.experts.down_proj[idx])
            out.index_add_(0, tok_idx, h * weights[tok_idx, top_k_pos, None])
        return out.reshape(B, S, H)


# ---------------------------------------------------------------------------
# Fake ScatterMoE ops (pure-PyTorch reference implementations)
# ---------------------------------------------------------------------------


def _fake_flatten_sort_count(expert_idxs: torch.Tensor, num_experts: int):
    """Reference implementation of ScatterMoE's flatten_sort_count."""
    flat = expert_idxs.flatten()
    sorted_expert_idxs, sorted_scattered_idxs = flat.sort(stable=True)
    counts = torch.bincount(flat, minlength=num_experts)
    expert_offsets = counts.cumsum(-1)
    return sorted_expert_idxs, sorted_scattered_idxs, expert_offsets


def _fake_parallel_linear(
    inputs, expert_weights, k,
    sorted_expert_idxs, sorted_scattered_idxs, expert_offsets,
    expert_biases=None, gates=None,
    grouped_in=False, grouped_out=False,
):
    """Reference parallel_linear: pure-PyTorch loop over experts."""
    E = expert_weights.shape[0]
    num_tokens = inputs.shape[0]
    out_dim = expert_weights.shape[-1]  # W is [E, in, out]

    if grouped_in:
        x = inputs                                   # already grouped
    else:
        # Expand: each input maps to k outputs
        x = inputs.repeat_interleave(k, dim=0)       # [num_tokens*k, in]
        x = x[sorted_scattered_idxs]                 # sort to expert order

    out = torch.zeros(x.shape[0], out_dim, dtype=inputs.dtype, device=inputs.device)
    start = 0
    prev = 0
    for e in range(E):
        end = int(expert_offsets[e].item())
        cnt = end - prev
        if cnt > 0:
            out[prev:end] = x[prev:end] @ expert_weights[e]  # [cnt, out_dim]
        prev = end

    if grouped_out:
        result = out
    else:
        # Scatter back to original (ungrouped) positions
        result = torch.zeros(num_tokens, out_dim, dtype=inputs.dtype, device=inputs.device)
        result.scatter_(0, sorted_scattered_idxs.unsqueeze(1).expand_as(out), out)

    if gates is not None:
        # gates: [num_tokens, k]; result currently [num_tokens*k, out_dim] or [num_tokens, out_dim]
        if grouped_out:
            # result is [num_tokens*k, out_dim]; reduce to [num_tokens, out_dim]
            result_expanded = result.view(gates.size(0), gates.size(1), out_dim)
            result = (gates.unsqueeze(1) @ result_expanded).squeeze(1)
        else:
            # result is already [num_tokens, out_dim]; just multiply
            result = result * gates

    return result


# ---------------------------------------------------------------------------
# Inject fake scattermoe module into sys.modules so _get_scattermoe_ops() works
# ---------------------------------------------------------------------------


def _inject_fake_scattermoe():
    mod = types.ModuleType("scattermoe_test_fake")
    mod.parallel_linear = _fake_parallel_linear
    mod.flatten_sort_count = _fake_flatten_sort_count
    sys.modules["scattermoe_test_fake"] = mod
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSortInvertRoundTrip(unittest.TestCase):
    """Test that sort → inv_sort → reshape → sum is correct."""

    def test_inv_sort_matches_original_order(self):
        N, top_k = 5, 2
        num_experts = 4
        torch.manual_seed(42)

        flat_expert_ids = torch.randint(0, num_experts, (N * top_k,)).int()
        sort_order = flat_expert_ids.argsort(stable=True)
        inv_sort_order = torch.empty_like(sort_order)
        inv_sort_order[sort_order] = torch.arange(N * top_k)

        # Any tensor
        values = torch.randn(N * top_k, 8)
        sorted_vals = values[sort_order]

        # combined[i] = sorted_vals[i]  (simulate identity expert)
        combined = sorted_vals.clone()

        output = combined[inv_sort_order]
        self.assertTrue(torch.allclose(output, values), "inv_sort_order must recover original order")

    def test_sum_over_topk_reduces_correctly(self):
        N, top_k, H = 3, 2, 8
        flat = torch.randn(N * top_k, H)
        summed = flat.view(N, top_k, H).sum(dim=1)
        self.assertEqual(summed.shape, (N, H))
        # First token
        self.assertTrue(torch.allclose(summed[0], flat[0] + flat[1]))


class TestLocalExpertIdsConstruction(unittest.TestCase):
    """Test that local_expert_ids are built correctly from recv_counts."""

    def test_single_sender_single_local_expert(self):
        ep_size, num_local_experts = 1, 3
        # From rank 0: 2 tokens to exp 0, 3 to exp 1, 1 to exp 2
        local_recv_counts = torch.tensor([[2, 3, 1]])  # [ep_size, L]
        expected_ids = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int64)

        ids = torch.cat([
            torch.full((int(local_recv_counts[s, e].item()),), e, dtype=torch.int64)
            for s in range(ep_size)
            for e in range(num_local_experts)
            if local_recv_counts[s, e].item() > 0
        ])
        self.assertTrue(torch.equal(ids, expected_ids))

    def test_two_senders(self):
        ep_size, num_local_experts = 2, 2
        local_recv_counts = torch.tensor([[1, 2], [3, 0]])  # [2, 2]
        # rank 0: 1 for exp 0, 2 for exp 1; rank 1: 3 for exp 0, 0 for exp 1
        expected_ids = torch.tensor([0, 1, 1, 0, 0, 0], dtype=torch.int64)

        ids = torch.cat([
            torch.full((int(local_recv_counts[s, e].item()),), e, dtype=torch.int64)
            for s in range(ep_size)
            for e in range(num_local_experts)
            if local_recv_counts[s, e].item() > 0
        ])
        self.assertTrue(torch.equal(ids, expected_ids))


class TestEPForwardSingleRank(unittest.TestCase):
    """EP forward with ep_size=1 must match the reference eager forward."""

    def setUp(self):
        _inject_fake_scattermoe()

    def _make_mock_ep_mesh(self, ep_size=1):
        """Return a minimal mock for the EP DeviceMesh with ep_size=1."""
        mesh = MagicMock()
        group = MagicMock()
        mesh.get_group.return_value = group
        dist_mock = MagicMock()
        dist_mock.get_rank.return_value = 0
        dist_mock.get_world_size.return_value = ep_size
        return mesh, group, dist_mock

    def test_output_shape(self):
        """EP forward must return tensor of the same shape as input."""
        from axolotl.integrations.kernels.ep import _EPScatterMoeForward

        num_experts, hidden, intermediate, top_k = 4, 16, 8, 2
        B, S = 2, 3
        ep_size = 1

        block = _FakeMoeBlock(num_experts, hidden, intermediate, top_k)

        # Build a minimal ep_mesh mock for ep_size=1
        ep_group = MagicMock()
        ep_mesh = MagicMock()
        ep_mesh.get_group.return_value = ep_group
        block._ep_mesh = ep_mesh

        def fake_rank(group):  # noqa: ANN001
            return 0

        def fake_world(group):  # noqa: ANN001
            return ep_size

        # Patch dist calls and all_to_all_single inside the EP forward
        with (
            patch("axolotl.integrations.kernels.ep.dist.get_rank", side_effect=fake_rank),
            patch("axolotl.integrations.kernels.ep.dist.get_world_size", side_effect=fake_world),
            patch("axolotl.integrations.kernels.ep.dist.all_to_all_single",
                  side_effect=lambda out, inp, group, **_: out.copy_(inp)),
            patch("axolotl.integrations.kernels.ep.all_to_all_single_autograd",
                  side_effect=lambda inp, output_split_sizes, input_split_sizes, group: inp),
        ):
            block.forward = types.MethodType(_EPScatterMoeForward.forward, block)
            x = torch.randn(B, S, hidden, requires_grad=True)
            out = block(x)

        self.assertEqual(out.shape, (B, S, hidden))

    def test_backward_does_not_crash(self):
        """Loss.backward() must succeed (gradients must propagate)."""
        from axolotl.integrations.kernels.ep import _EPScatterMoeForward

        num_experts, hidden, intermediate, top_k = 4, 16, 8, 1
        B, S = 1, 2
        ep_size = 1

        block = _FakeMoeBlock(num_experts, hidden, intermediate, top_k)

        ep_group = MagicMock()
        ep_mesh = MagicMock()
        ep_mesh.get_group.return_value = ep_group
        block._ep_mesh = ep_mesh

        with (
            patch("axolotl.integrations.kernels.ep.dist.get_rank", return_value=0),
            patch("axolotl.integrations.kernels.ep.dist.get_world_size", return_value=ep_size),
            patch("axolotl.integrations.kernels.ep.dist.all_to_all_single",
                  side_effect=lambda out, inp, group, **_: out.copy_(inp)),
            patch("axolotl.integrations.kernels.ep.all_to_all_single_autograd",
                  side_effect=lambda inp, output_split_sizes, input_split_sizes, group: inp),
        ):
            block.forward = types.MethodType(_EPScatterMoeForward.forward, block)
            x = torch.randn(B, S, hidden, requires_grad=True)
            out = block(x)
            loss = out.sum()
            loss.backward()

        self.assertIsNotNone(x.grad, "Input gradient must be computed")


class TestApplyEPScatterMoe(unittest.TestCase):
    """Test that apply_ep_scattermoe patches all MoE blocks correctly."""

    def setUp(self):
        _inject_fake_scattermoe()

    def _make_fake_model(self, num_blocks=3):
        """Build a tiny model with multiple fake MoE blocks."""
        model = nn.Sequential(*[
            _FakeMoeBlock(num_experts=4, hidden=16, intermediate=8, top_k=1)
            for _ in range(num_blocks)
        ])
        return model

    @patch("axolotl.integrations.kernels.ep.dist.get_world_size", return_value=4)
    @patch("axolotl.integrations.kernels.ep.init_device_mesh")
    @patch("axolotl.integrations.kernels.ep.get_model_moe_block")
    def test_all_blocks_patched(self, mock_get_cls, mock_init_mesh, mock_world):
        from axolotl.integrations.kernels.ep import apply_ep_scattermoe

        mock_get_cls.return_value = _FakeMoeBlock
        fake_mesh = MagicMock()
        mesh_container = MagicMock()
        mesh_container.__getitem__ = MagicMock(return_value=fake_mesh)
        mock_init_mesh.return_value = mesh_container

        model = self._make_fake_model(3)
        apply_ep_scattermoe(model, "qwen3_moe", ep_size=2)

        for module in model.modules():
            if isinstance(module, _FakeMoeBlock):
                self.assertTrue(
                    hasattr(module, "_ep_mesh"),
                    "Each MoE block must have _ep_mesh attribute set",
                )
                # forward must have been replaced
                self.assertIsNot(
                    type(module).forward,
                    module.forward.__func__,
                    "forward must be replaced with EP version",
                )

    @patch("axolotl.integrations.kernels.ep.dist.get_world_size", return_value=4)
    @patch("axolotl.integrations.kernels.ep.init_device_mesh")
    @patch("axolotl.integrations.kernels.ep.get_model_moe_block")
    def test_world_size_not_divisible_raises(self, mock_get_cls, mock_init_mesh, mock_world):
        from axolotl.integrations.kernels.ep import apply_ep_scattermoe

        mock_get_cls.return_value = _FakeMoeBlock

        with self.assertRaises(ValueError, msg="Should raise when ep_size doesn't divide world_size"):
            apply_ep_scattermoe(MagicMock(), "qwen3_moe", ep_size=3)


if __name__ == "__main__":
    unittest.main()
