import mlx.core as mx
import mlx.nn as nn

from ..qwen3_5 import Model as Qwen3_5Model
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


class Model(Qwen3_5Model):

    def __init__(self, config: ModelConfig):
        # only initialize nn.Module, skip the initialization of vision_tower and language_model in the parent class
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def sanitize(self, weights):
        if self.config.text_config.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        for l in range(self.config.text_config.num_hidden_layers):
            prefix = f"model.language_model.layers.{l}.mlp"
            # process gate_up_proj [num_experts, 2 * intermediate_size, hidden_size]
            gate_up_weight = weights.pop(f"{prefix}.experts.gate_up_proj")
            gate_weight, up_weights = mx.split(gate_up_weight, 2, axis=-2)
            weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_weight
            weights[f"{prefix}.switch_mlp.up_proj.weight"] = up_weights
            # down_proj
            weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                f"{prefix}.experts.down_proj"
            )

        mtp_num = self.config.text_config.mtp_num_hidden_layers
        num_experts = self.config.text_config.num_experts
        for layer_idx in range(mtp_num):
            prefix = f"mtp.layers.{layer_idx}.mlp"
            gate_up = weights.pop(f"{prefix}.experts.gate_up_proj", None)
            if gate_up is not None:
                # Fused format (Qwen3.6): gate_up_proj [num_experts, 2*intermediate, hidden]
                mid = gate_up.shape[-2] // 2
                weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :mid, :]
                weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., mid:, :]
                weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
                    f"{prefix}.experts.down_proj"
                )
            elif f"{prefix}.experts.0.gate_proj.weight" in weights:
                # Per-expert format (Qwen3.5): stack individual expert weights
                for n in ["gate_proj", "up_proj", "down_proj"]:
                    stacked = mx.stack(
                        [
                            weights.pop(f"{prefix}.experts.{e}.{n}.weight")
                            for e in range(num_experts)
                        ]
                    )
                    weights[f"{prefix}.switch_mlp.{n}.weight"] = stacked

        norm_keys = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".pre_fc_norm_hidden.weight",
            ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        )

        sanitized_weights = {}
        for key, value in weights.items():
            if "model" in key:
                if "model.language_model" in key:
                    key = key.replace("model.language_model", "language_model.model")
                elif "model.visual" in key:
                    key = key.replace("model.visual", "vision_tower")
            elif "lm_head" in key:
                key = key.replace("lm_head", "language_model.lm_head")
            elif key.startswith("mtp."):
                key = f"language_model.{key}"

            if "conv1d.weight" in key and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            if any(key.endswith(sfx) for sfx in norm_keys):
                if value.ndim == 1:
                    value += 1.0

            sanitized_weights[key] = value

        return sanitized_weights
