#--- START OF MODIFIED FILE app_fluxContext_Ltx/ltx_video/models/transformers/transformer3d.py ---
# Adapted from: https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/models/transformers/transformer_2d.py
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import os
import json
import glob
from pathlib import Path

import torch
import numpy as np
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.utils import BaseOutput, is_torch_version
from diffusers.utils import logging
from torch import nn
from safetensors import safe_open


from ltx_video.models.transformers.attention import BasicTransformerBlock
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

from ltx_video.utils.diffusers_config_mapping import (
    diffusers_and_ours_config_mapping,
    make_hashable_key,
    TRANSFORMER_KEYS_RENAME_DICT,
)


logger = logging.get_logger(__name__)


@dataclass
class Transformer3DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


class Transformer3DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        num_vector_embeds: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        adaptive_norm: str = "single_scale_shift",  # 'single_scale_shift' or 'single_scale'
        standardization_norm: str = "layer_norm",  # 'layer_norm' or 'rms_norm'
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        use_tpu_flash_attention: bool = False,  # if True uses the TPU attention offload ('flash attention')
        qk_norm: Optional[str] = None,
        positional_embedding_type: str = "rope",
        positional_embedding_theta: Optional[float] = None,
        positional_embedding_max_pos: Optional[List[int]] = None,
        timestep_scale_multiplier: Optional[float] = None,
        causal_temporal_positioning: bool = False,  # For backward compatibility, will be deprecated
    ):
        super().__init__()
        self.use_tpu_flash_attention = (
            use_tpu_flash_attention  # FIXME: push config down to the attention modules
        )
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.patchify_proj = nn.Linear(in_channels, inner_dim, bias=True)
        self.positional_embedding_type = positional_embedding_type
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos
        self.use_rope = self.positional_embedding_type == "rope"
        self.timestep_scale_multiplier = timestep_scale_multiplier

        if self.positional_embedding_type == "absolute":
            raise ValueError("Absolute positional embedding is no longer supported")
        elif self.positional_embedding_type == "rope":
            if positional_embedding_theta is None:
                raise ValueError(
                    "If `positional_embedding_type` type is rope, `positional_embedding_theta` must also be defined"
                )
            if positional_embedding_max_pos is None:
                raise ValueError(
                    "If `positional_embedding_type` type is rope, `positional_embedding_max_pos` must also be defined"
                )

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    adaptive_norm=adaptive_norm,
                    standardization_norm=standardization_norm,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    attention_type=attention_type,
                    use_tpu_flash_attention=use_tpu_flash_attention,
                    qk_norm=qk_norm,
                    use_rope=self.use_rope,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, inner_dim) / inner_dim**0.5
        )
        self.proj_out = nn.Linear(inner_dim, self.out_channels)

        self.adaln_single = AdaLayerNormSingle(
            inner_dim, use_additional_conditions=False
        )
        if adaptive_norm == "single_scale":
            self.adaln_single.linear = nn.Linear(inner_dim, 4 * inner_dim, bias=True)

        self.caption_projection = None
        if caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels, hidden_size=inner_dim
            )

        self.gradient_checkpointing = False

    def set_use_tpu_flash_attention(self):
        r"""
        Function sets the flag in this object and propagates down the children. The flag will enforce the usage of TPU
        attention kernel.
        """
        logger.info("ENABLE TPU FLASH ATTENTION -> TRUE")
        self.use_tpu_flash_attention = True
        # push config down to the attention modules
        for block in self.transformer_blocks:
            block.set_use_tpu_flash_attention()

    def create_skip_layer_mask(
        self,
        batch_size: int,
        num_conds: int,
        ptb_index: int,
        skip_block_list: Optional[List[int]] = None,
    ):
        if skip_block_list is None or len(skip_block_list) == 0:
            return None
        num_layers = len(self.transformer_blocks)
        mask = torch.ones(
            (num_layers, batch_size * num_conds), device=self.device, dtype=self.dtype
        )
        for block_idx in skip_block_list:
            mask[block_idx, ptb_index::num_conds] = 0
        return mask

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def get_fractional_positions(self, indices_grid):
        fractional_positions = torch.stack(
            [
                indices_grid[:, i] / self.positional_embedding_max_pos[i]
                for i in range(3)
            ],
            dim=-1,
        )
        return fractional_positions

    def precompute_freqs_cis(self, indices_grid, spacing="exp"):
        dtype = torch.float32  # We need full precision in the freqs_cis computation.
        dim = self.inner_dim
        theta = self.positional_embedding_theta

        fractional_positions = self.get_fractional_positions(indices_grid)

        start = 1
        end = theta
        device = fractional_positions.device
        if spacing == "exp":
            indices = theta ** (
                torch.linspace(
                    math.log(start, theta),
                    math.log(end, theta),
                    dim // 6,
                    device=device,
                    dtype=dtype,
                )
            )
            indices = indices.to(dtype=dtype)
        elif spacing == "exp_2":
            indices = 1.0 / theta ** (torch.arange(0, dim, 6, device=device) / dim)
            indices = indices.to(dtype=dtype)
        elif spacing == "linear":
            indices = torch.linspace(start, end, dim // 6, device=device, dtype=dtype)
        elif spacing == "sqrt":
            indices = torch.linspace(
                start**2, end**2, dim // 6, device=device, dtype=dtype
            ).sqrt()

        indices = indices * math.pi / 2

        if spacing == "exp_2":
            freqs = (
                (indices * fractional_positions.unsqueeze(-1))
                .transpose(-1, -2)
                .flatten(2)
            )
        else:
            freqs = (
                (indices * (fractional_positions.unsqueeze(-1) * 2 - 1))
                .transpose(-1, -2)
                .flatten(2)
            )

        cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freq = freqs.sin().repeat_interleave(2, dim=-1)
        if dim % 6 != 0:
            cos_padding = torch.ones_like(cos_freq[:, :, : dim % 6])
            sin_padding = torch.zeros_like(cos_freq[:, :, : dim % 6])
            cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
            sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)
        return cos_freq.to(self.dtype), sin_freq.to(self.dtype)

    def load_state_dict(
        self,
        state_dict: Dict,
        *args,
        **kwargs,
    ):
        if any([key.startswith("model.diffusion_model.") for key in state_dict.keys()]):
            state_dict = {
                key.replace("model.diffusion_model.", ""): value
                for key, value in state_dict.items()
                if key.startswith("model.diffusion_model.")
            }
        super().load_state_dict(state_dict, *args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: Optional[Union[str, os.PathLike]],
        *args,
        **kwargs,
    ):
        pretrained_model_path = Path(pretrained_model_path)
        if pretrained_model_path.is_dir():
            config_path = pretrained_model_path / "transformer" / "config.json"
            with open(config_path, "r") as f:
                config = make_hashable_key(json.load(f))

            assert config in diffusers_and_ours_config_mapping, (
                "Provided diffusers checkpoint config for transformer is not suppported. "
                "We only support diffusers configs found in Lightricks/LTX-Video."
            )

            config = diffusers_and_ours_config_mapping[config]
            state_dict = {}
            ckpt_paths = (
                pretrained_model_path
                / "transformer"
                / "diffusion_pytorch_model*.safetensors"
            )
            dict_list = glob.glob(str(ckpt_paths))
            for dict_path in dict_list:
                part_dict = {}
                with safe_open(dict_path, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        part_dict[k] = f.get_tensor(k)
                state_dict.update(part_dict)

            for key in list(state_dict.keys()):
                new_key = key
                for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
                    new_key = new_key.replace(replace_key, rename_key)
                state_dict[new_key] = state_dict.pop(key)

            with torch.device("meta"):
                transformer = cls.from_config(config)
            transformer.load_state_dict(state_dict, assign=True, strict=True)
        elif pretrained_model_path.is_file() and str(pretrained_model_path).endswith(
            ".safetensors"
        ):
            comfy_single_file_state_dict = {}
            with safe_open(pretrained_model_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                for k in f.keys():
                    comfy_single_file_state_dict[k] = f.get_tensor(k)
            configs = json.loads(metadata["config"])
            transformer_config = configs["transformer"]
            with torch.device("meta"):
                transformer = Transformer3DModel.from_config(transformer_config)
            transformer.load_state_dict(comfy_single_file_state_dict, assign=True)
        return transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        indices_grid: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        skip_layer_mask: Optional[torch.Tensor] = None,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
        return_dict: bool = True,
    ):
        if not self.use_tpu_flash_attention:
            if attention_mask is not None and attention_mask.ndim == 2:
                attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                encoder_attention_mask = (
                    1 - encoder_attention_mask.to(hidden_states.dtype)
                ) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        hidden_states = self.patchify_proj(hidden_states)

        if self.timestep_scale_multiplier:
            timestep = self.timestep_scale_multiplier * timestep

        freqs_cis = self.precompute_freqs_cis(indices_grid)

        batch_size = hidden_states.shape[0]
        timestep, embedded_timestep = self.adaln_single(
            timestep.flatten(),
            {"resolution": None, "aspect_ratio": None},
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.shape[-1]
        )
        
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
            )

        # TeaCache Integration
        if hasattr(self, 'enable_teacache') and self.enable_teacache:
            ori_hidden_states = hidden_states.clone()
            temb_ = embedded_timestep.clone()
            inp = self.transformer_blocks[0].norm1(hidden_states.clone())
            
            first_block = self.transformer_blocks[0]
            modulated_inp = inp
            if first_block.adaptive_norm in ["single_scale_shift", "single_scale"]:
                num_ada_params = first_block.scale_shift_table.shape[0]
                ada_values = first_block.scale_shift_table[None, None] + temb_.reshape(
                    batch_size, temb_.shape[1], num_ada_params, -1
                )
                if first_block.adaptive_norm == "single_scale_shift":
                    shift_msa, scale_msa, _, _, _, _ = ada_values.unbind(dim=2)
                    modulated_inp = inp * (1 + scale_msa) + shift_msa
                else:
                    scale_msa, _, _, _ = ada_values.unbind(dim=2)
                    modulated_inp = inp * (1 + scale_msa)

            should_calc = False
            if self.cnt == 0 or self.cnt == self.num_steps - 1 or self.previous_modulated_input is None:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else:
                coefficients = [2.14700694e+01, -1.28016453e+01,  2.31279151e+00,  7.92487521e-01, 9.69274326e-03]
                rescale_func = np.poly1d(coefficients)
                rel_l1_dist = ((modulated_inp - self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item()
                self.accumulated_rel_l1_distance += rescale_func(rel_l1_dist)
                
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            
            self.previous_modulated_input = modulated_inp
            self.cnt += 1
            if self.cnt == self.num_steps:
                self.cnt = 0
            
            if not should_calc and self.previous_residual is not None:
                hidden_states = ori_hidden_states + self.previous_residual
            else:
                # Execute original logic if cache is missed
                temp_hidden_states = hidden_states
                for block_idx, block in enumerate(self.transformer_blocks):
                    temp_hidden_states = block(
                        temp_hidden_states,
                        freqs_cis=freqs_cis,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        timestep=timestep,
                        cross_attention_kwargs=cross_attention_kwargs,
                        class_labels=class_labels,
                        skip_layer_mask=(skip_layer_mask[block_idx] if skip_layer_mask is not None else None),
                        skip_layer_strategy=skip_layer_strategy,
                    )
                self.previous_residual = temp_hidden_states - ori_hidden_states
                hidden_states = temp_hidden_states
        else:
            # Original path if TeaCache is disabled
            for block_idx, block in enumerate(self.transformer_blocks):
                hidden_states = block(
                    hidden_states,
                    freqs_cis=freqs_cis,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    skip_layer_mask=(skip_layer_mask[block_idx] if skip_layer_mask is not None else None),
                    skip_layer_strategy=skip_layer_strategy,
                )

        # Final modulation and output
        scale_shift_values = (self.scale_shift_table[None, None] + embedded_timestep[:, :, None])
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)

        if not return_dict:
            return (hidden_states,)

        return Transformer3DModelOutput(sample=hidden_states)
#--- END OF MODIFIED FILE app_fluxContext_Ltx/ltx_video/models/transformers/transformer3d.py ---