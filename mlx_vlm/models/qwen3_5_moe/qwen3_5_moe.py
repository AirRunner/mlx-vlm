import mlx.core as mx
import mlx.nn as nn

from ..qwen3_5 import Model as Qwen3_5Model
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel


def _unfuse_experts(weights, prefix):
    """Split fused gate_up_proj into per-projection switch_mlp weights (Qwen3.6 format)."""
    gate_up_key = f"{prefix}.experts.gate_up_proj"
    if gate_up_key not in weights:
        return
    gate_up = weights.pop(gate_up_key)
    mid = gate_up.shape[-2] // 2
    weights[f"{prefix}.switch_mlp.gate_proj.weight"] = gate_up[..., :mid, :]
    weights[f"{prefix}.switch_mlp.up_proj.weight"] = gate_up[..., mid:, :]
    weights[f"{prefix}.switch_mlp.down_proj.weight"] = weights.pop(
        f"{prefix}.experts.down_proj"
    )


def _stack_per_expert(weights, prefix, num_experts):
    """Stack per-expert weights into switch_mlp format (Qwen3.5 format)."""
    for n in ("gate_proj", "up_proj", "down_proj"):
        weights[f"{prefix}.switch_mlp.{n}.weight"] = mx.stack(
            [
                weights.pop(f"{prefix}.experts.{e}.{n}.weight")
                for e in range(num_experts)
            ]
        )


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

        # Backbone MoE layers always use fused gate_up_proj.
        for l in range(self.config.text_config.num_hidden_layers):
            _unfuse_experts(weights, f"model.language_model.layers.{l}.mlp")

        # MTP layers: fused format (Qwen3.6) or per-expert format (Qwen3.5).
        # Detect format once from the first layer and apply uniformly.
        mtp_num = self.config.text_config.mtp_num_hidden_layers
        if mtp_num > 0:
            num_experts = self.config.text_config.num_experts
            mtp_is_fused = "mtp.layers.0.mlp.experts.gate_up_proj" in weights
            for layer_idx in range(mtp_num):
                prefix = f"mtp.layers.{layer_idx}.mlp"
                if mtp_is_fused:
                    _unfuse_experts(weights, prefix)
                else:
                    _stack_per_expert(weights, prefix, num_experts)

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
