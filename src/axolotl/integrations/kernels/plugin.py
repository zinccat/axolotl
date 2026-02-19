from kernels import (
    LayerRepository,
    Mode,
    register_kernel_mapping,
    replace_kernel_forward_from_hub,
)

from axolotl.integrations.base import BasePlugin


class KernelsPlugin(BasePlugin):
    def get_input_args(self):
        return "axolotl.integrations.kernels.KernelsArgs"

    def pre_model_load(self, cfg):
        if cfg.use_scattermoe:
            self._register_kernels()
            ep_size = getattr(cfg, "expert_parallel_size", None)
            if not ep_size or ep_size <= 1:
                # Single-rank ScatterMoE: patch the block class so that the
                # kernels library swaps the forward when use_kernels=True loads.
                self._kernelize_model(cfg.model_config_type)
            # For EP > 1: block-level patching happens in post_model_load once
            # the model instances exist and distributed is initialised.

    def post_model_load(self, cfg, model):
        ep_size = getattr(cfg, "expert_parallel_size", None)
        if getattr(cfg, "use_scattermoe", False) and ep_size and ep_size > 1:
            from axolotl.integrations.kernels.ep import apply_ep_scattermoe

            apply_ep_scattermoe(model, cfg.model_config_type, ep_size)

    def _register_kernels(self):
        register_kernel_mapping(
            {
                "HFScatterMoEParallelExperts": {
                    "cuda": {
                        Mode.TRAINING: LayerRepository(
                            repo_id="axolotl-ai-co/scattermoe",
                            layer_name="HFScatterMoEGatedMLP",
                        ),
                        Mode.INFERENCE: LayerRepository(
                            repo_id="axolotl-ai-co/scattermoe",
                            layer_name="HFScatterMoEGatedMLP",
                        ),
                    },
                }
            }
        )

    def _kernelize_model(self, model_type: str):
        from axolotl.integrations.kernels.ep import get_model_moe_block

        if model_type == "olmoe":
            from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

            replace_kernel_forward_from_hub(
                OlmoeSparseMoeBlock, "HFScatterMoEParallelExperts"
            )
        else:
            try:
                model_moe_cls = get_model_moe_block(model_type)
                replace_kernel_forward_from_hub(
                    model_moe_cls, "HFScatterMoEParallelExperts"
                )
            except Exception as err:
                raise ValueError(f"Unsupported model type: {model_type}") from err
