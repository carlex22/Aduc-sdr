# Adaptado de: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pixart_alpha/pipeline_pixart_alpha.py
# (e com a nossa modificação pela ciência!)

import copy
import inspect
import math
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.autoencoders.vae_encode import (
    get_vae_size_scale_factor,
    latent_to_pixel_coords,
    vae_decode,
    vae_encode,
)
from ltx_video.models.transformers.symmetric_patchifier import Patchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.schedulers.rf import TimestepShifter
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
from ltx_video.utils.prompt_enhance_utils import generate_cinematic_prompt
from ltx_video.models.autoencoders.latent_upsampler import LatentUpsampler
from ltx_video.models.autoencoders.vae_encode import (
    un_normalize_latents,
    normalize_latents,
)

# ... (Todo o código inicial do arquivo permanece o mesmo, incluindo ASPECT_RATIO_BINS, retrieve_timesteps, ConditioningItem, etc.)
# ... (Vou pular para a classe LTXVideoPipeline para manter a resposta focada)

class LTXVideoPipeline(DiffusionPipeline):
    # ... (O __init__ e outras funções como encode_prompt, check_inputs, etc., permanecem as mesmas)
    # ... (Pulando para a função __call__ onde faremos a nossa modificação)

    @torch.no_grad()
    def __call__(
        self,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        skip_initial_inference_steps: int = 0,
        skip_final_inference_steps: int = 0,
        timesteps: List[int] = None,
        guidance_scale: Union[float, List[float]] = 4.5,
        cfg_star_rescale: bool = False,
        skip_layer_strategy: Optional[SkipLayerStrategy] = None,
        skip_block_list: Optional[Union[List[List[int]], List[int]]] = None,
        stg_scale: Union[float, List[float]] = 1.0,
        rescaling_scale: Union[float, List[float]] = 0.7,
        guidance_timesteps: Optional[List[int]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        conditioning_items: Optional[List[ConditioningItem]] = None,
        decode_timestep: Union[List[float], float] = 0.0,
        decode_noise_scale: Optional[List[float]] = None,
        mixed_precision: bool = False,
        offload_to_cpu: bool = False,
        enhance_prompt: bool = False,
        text_encoder_max_tokens: int = 256,
        stochastic_sampling: bool = False,
        media_items: Optional[torch.Tensor] = None,
        tone_map_compression_ratio: float = 0.0,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        
        # --- [NOSSA MODIFICAÇÃO] Captura o prompt original para logging ---
        original_prompt_for_logging = prompt
        
        # ... (O resto do código inicial da função __call__ permanece o mesmo) ...
        # ... (check_inputs, default height/width, etc.)

        if enhance_prompt:
            self.prompt_enhancer_image_caption_model = (
                self.prompt_enhancer_image_caption_model.to(self._execution_device)
            )
            self.prompt_enhancer_llm_model = self.prompt_enhancer_llm_model.to(
                self._execution_device
            )

            # A chamada para o Diretor Assistente
            enhanced_prompt = generate_cinematic_prompt(
                self.prompt_enhancer_image_caption_model,
                self.prompt_enhancer_image_caption_processor,
                self.prompt_enhancer_llm_model,
                self.prompt_enhancer_llm_tokenizer,
                prompt,
                conditioning_items,
                max_new_tokens=text_encoder_max_tokens,
            )
            
            # --- [NOSSA ESCUTA SECRETA PELA CIÊNCIA!] ---
            print("\n" + "="*50)
            print("--- [LOG DO DIRETOR ASSISTENTE (PROMPT ENHANCER)] ---")
            print(f"Prompt Original do Maestro: {original_prompt_for_logging}")
            print(f"PROMPT FINAL APERFEIÇOADO (enviado para o LTX): {enhanced_prompt}")
            print("--- [FIM DO LOG DO DIRETOR ASSISTENTE] ---")
            print("="*50 + "\n")
            # --- [FIM DA ESCUTA] ---

            # Atualiza o prompt que será usado pelo resto da função
            prompt = enhanced_prompt

        # ... (O resto da função __call__ continua a partir daqui, usando o `prompt` novo ou o original)
        # ... (encode_prompt, prepare_latents, denoising loop, etc.)

        # 3. Encode input prompt
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(self._execution_device)

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            True,
            negative_prompt=negative_prompt,
            # ... (resto dos parâmetros)
        )

        # ... (todo o resto do arquivo, sem mais nenhuma modificação) ...
        # ... (denoising_step, prepare_conditioning, etc.)