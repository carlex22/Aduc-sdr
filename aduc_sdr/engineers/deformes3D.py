# engineers/deformes3D.py
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
# Version 1.4.5

from PIL import Image
import os
import time
import logging
import gradio as gr
import yaml

from managers.flux_kontext_manager import flux_kontext_singleton
from engineers.deformes2D_thinker import deformes2d_thinker_singleton

logger = logging.getLogger(__name__)

class Deformes3DEngine:
    """
    ADUC Specialist for static image (keyframe) generation.
    This is responsible for the entire process of turning a script into a gallery of keyframes.
    """
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir
        self.image_generation_helper = flux_kontext_singleton
        logger.info("3D Engine (Image Specialist) ready to receive orders from the Maestro.")

    def _generate_single_keyframe(self, prompt: str, reference_images: list[Image.Image], output_filename: str, width: int, height: int, callback: callable = None) -> str:
        """
        Low-level function that generates a single image.
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
        Orchestrates the generation of all keyframes from a storyboard.
        """
        current_base_image_path = initial_ref_path
        previous_prompt = "N/A (initial reference image)"
        final_keyframes = [current_base_image_path]
        width, height = keyframe_resolution, keyframe_resolution
        
        num_keyframes_to_generate = len(storyboard) - 1
        
        logger.info(f"IMAGE SPECIALIST: Received order to generate {num_keyframes_to_generate} keyframes.")

        for i in range(num_keyframes_to_generate):
            current_scene = storyboard[i]
            future_scene = storyboard[i+1]
            progress_callback = progress_callback_factory(i + 1, num_keyframes_to_generate) if progress_callback_factory else None
            
            logger.info(f"--> Generating Keyframe {i+1}/{num_keyframes_to_generate}...")
            
            new_flux_prompt = deformes2d_thinker_singleton.get_anticipatory_keyframe_prompt(
                global_prompt=global_prompt, scene_history=previous_prompt,
                current_scene_desc=current_scene, future_scene_desc=future_scene,
                last_image_path=current_base_image_path, fixed_ref_paths=general_ref_paths
            )
            
            images_for_flux_paths = list(set([current_base_image_path] + general_ref_paths))
            images_for_flux = [Image.open(p) for p in images_for_flux_paths]
            
            new_keyframe_path = self._generate_single_keyframe(
                prompt=new_flux_prompt, reference_images=images_for_flux,
                output_filename=f"keyframe_{i+1}.png", width=width, height=height,
                callback=progress_callback
            )

            final_keyframes.append(new_keyframe_path)
            current_base_image_path = new_keyframe_path
            previous_prompt = new_flux_prompt
            
        logger.info(f"IMAGE SPECIALIST: Keyframe generation complete.")
        return final_keyframes

# --- Singleton Instantiation ---
try:
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    WORKSPACE_DIR = config['application']['workspace_dir']
    
    # Correctly instantiate the Deformes3DEngine class
    deformes3d_engine_singleton = Deformes3DEngine(workspace_dir=WORKSPACE_DIR)
    
except Exception as e:
    logger.error(f"Could not initialize Deformes3DEngine: {e}", exc_info=True)
    deformes3d_engine_singleton = None