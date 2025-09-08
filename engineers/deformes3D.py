# engineers/deformes3D.py
#
# AducSdr: Uma implementação aberta e funcional da arquitetura ADUC-SDR
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos
#
# Contato:
# Carlos Rodrigues dos Santos
# carlex22@gmail.com
# Rua Eduardo Carlos Pereira, 4125, B1 Ap32, Curitiba, PR, Brazil, CEP 8102025
#
# Repositórios e Projetos Relacionados:
# GitHub: https://github.com/carlex22/Aduc-sdr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License...
# PENDING PATENT NOTICE: Please see NOTICE.md.
#
# Version 2.0.1

from PIL import Image, ImageOps
import os
import time
import logging
import gradio as gr
import yaml
import torch
import numpy as np

from managers.flux_kontext_manager import flux_kontext_singleton
from engineers.deformes2D_thinker import deformes2d_thinker_singleton
from aduc_types import LatentConditioningItem
from managers.ltx_manager import ltx_manager_singleton
from managers.vae_manager import vae_manager_singleton
from managers.latent_enhancer_manager import latent_enhancer_specialist_singleton

logger = logging.getLogger(__name__)

class Deformes3DEngine:
    """
    ADUC Specialist for static image (keyframe) generation.
    """
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir
        self.image_generation_helper = flux_kontext_singleton
        logger.info("3D Engine (Image Specialist) ready to receive orders from the Maestro.")

    def _generate_single_keyframe(self, prompt: str, reference_images: list[Image.Image], output_filename: str, width: int, height: int, callback: callable = None) -> str:
        """
        Low-level function that generates a single image using the LTX helper.
        """
        logger.info(f"Generating keyframe '{output_filename}' with prompt: '{prompt}'")
        generated_image = self.image_generation_helper.generate_image(
            reference_images=reference_images, prompt=prompt, width=width,
            height=height, seed=int(time.time()), callback=callback
        )
        final_path = os.path.join(self.workspace_dir, output_filename)
        generated_image.save(final_path)
        logger.info(f"Keyframe successfully saved to: {final_path}")
        return final_path

    def generate_keyframes_from_storyboard(self, storyboard: list, initial_ref_path: str, global_prompt: str, keyframe_resolution: int, general_ref_paths: list, progress_callback_factory: callable = None):
        """
        Orchestrates the generation of all keyframes. 
        """
        current_base_image_path = initial_ref_path
        previous_prompt = "N/A (initial reference image)"
        final_keyframes_gallery = [] #[current_base_image_path]
        width, height = keyframe_resolution, keyframe_resolution
        target_resolution_tuple = (width, height)
        
        num_keyframes_to_generate = len(storyboard) - 1
        logger.info(f"IMAGE SPECIALIST: Received order to generate {num_keyframes_to_generate} keyframes (LTX versions).")

        for i in range(num_keyframes_to_generate):
            scene_index = i + 1
            current_scene = storyboard[i]
            future_scene = storyboard[i+1]
            progress_callback_flux = progress_callback_factory(scene_index, num_keyframes_to_generate) if progress_callback_factory else None
            
            logger.info(f"--> Generating Keyframe {scene_index}/{num_keyframes_to_generate}...")

            # --- STEP A: Generate with FLUX (Primary Method) ---
            logger.info(f"    - Step A: Generating with keyframe...")
            
            img_prompt = deformes2d_thinker_singleton.get_anticipatory_keyframe_prompt(
                global_prompt=global_prompt, scene_history=previous_prompt,
                current_scene_desc=current_scene, future_scene_desc=future_scene,
                last_image_path=current_base_image_path, fixed_ref_paths=general_ref_paths
            )
            
            #flux_ref_paths = list(set([current_base_image_path] + general_ref_paths))
            #flux_ref_images = [Image.open(p) for p in flux_ref_paths]
            
            #flux_keyframe_path = self._generate_single_keyframe(
            #    prompt=img_prompt, reference_images=flux_ref_images,
            #    output_filename=f"keyframe_{scene_index}_flux.png", width=width, height=height,
            #    callback=progress_callback_flux
            #)
            #final_keyframes_gallery.append(flux_keyframe_path)
            
            # --- STEP B: LTX Enrichment Experiment ---
            #logger.info(f"    - Step B: Generating enrichment with LTX...")

            ltx_context_paths = []
            context_paths = []
            context_paths = [current_base_image_path] + [p for p in general_ref_paths if p != current_base_image_path][:3]
            
            ltx_context_paths = list(reversed(context_paths))
            logger.info(f"    - LTX Context Order (Reversed): {[os.path.basename(p) for p in ltx_context_paths]}")

            ltx_conditioning_items = []
            
            weight = 0.6
            for idx, path in enumerate(ltx_context_paths):
                img_pil = Image.open(path).convert("RGB")
                img_processed = self._preprocess_image_for_latent_conversion(img_pil, target_resolution_tuple)
                pixel_tensor = self._pil_to_pixel_tensor(img_processed)
                latent_tensor = vae_manager_singleton.encode(pixel_tensor)
                
                ltx_conditioning_items.append(LatentConditioningItem(latent_tensor, 0, weight))
                
                if idx >= 0:
                    weight -= 0.1
            
            ltx_base_params = {"guidance_scale": 1.0, "stg_scale": 0.001, "num_inference_steps": 25}
            generated_latents, _ = ltx_manager_singleton.generate_latent_fragment(
                callback_on_step_end=progress_callback_flux
                height=height, width=width,
                conditioning_items_data=ltx_conditioning_items,
                motion_prompt=img_prompt,
                video_total_frames=48,
                video_fps=24,
                **ltx_base_params
            )

            final_latent = generated_latents[:, :, -1:, :, :]
            upscaled_latent = latent_enhancer_specialist_singleton.upscale(final_latent)
            enriched_pixel_tensor = vae_manager_singleton.decode(upscaled_latent)

            ltx_keyframe_path = os.path.join(self.workspace_dir, f"keyframe_{scene_index}_ltx.png")
            self.save_image_from_tensor(enriched_pixel_tensor, ltx_keyframe_path)
            final_keyframes_gallery.append(ltx_keyframe_path)
            
            # Use the FLUX keyframe as the base for the next iteration to maintain the primary narrative path
            current_base_image_path = ltx_keyframe_path #flux_keyframe_path 
            previous_prompt = img_prompt

        logger.info(f"IMAGE SPECIALIST: Generation of all keyframe versions (LTX) complete.")
        return final_keyframes_gallery

    # --- HELPER FUNCTIONS ---

    def _preprocess_image_for_latent_conversion(self, image: Image.Image, target_resolution: tuple) -> Image.Image:
        """Resizes and fits an image to the target resolution for VAE encoding."""
        if image.size != target_resolution:
            return ImageOps.fit(image, target_resolution, Image.Resampling.LANCZOS)
        return image
        
    def _pil_to_pixel_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Helper to convert PIL to the 5D pixel tensor the VAE expects."""
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        return (tensor * 2.0) - 1.0

    def save_image_from_tensor(self, pixel_tensor: torch.Tensor, path: str):
        """Helper to save a 1-frame pixel tensor as an image."""
        tensor_chw = pixel_tensor.squeeze(0).squeeze(1)
        tensor_hwc = tensor_chw.permute(1, 2, 0)
        tensor_hwc = (tensor_hwc.clamp(-1, 1) + 1) / 2.0
        image_np = (tensor_hwc.cpu().float().numpy() * 255).astype(np.uint8)
        Image.fromarray(image_np).save(path)

# --- Singleton Instantiation ---
try:
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    WORKSPACE_DIR = config['application']['workspace_dir']
    deformes3d_engine_singleton = Deformes3DEngine(workspace_dir=WORKSPACE_DIR)
except Exception as e:
    logger.error(f"Could not initialize Deformes3DEngine: {e}", exc_info=True)
    deformes3d_engine_singleton = None