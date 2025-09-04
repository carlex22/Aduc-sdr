# engineers/deformes2D_thinker.py
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
# Version 1.0.1

import logging
from pathlib import Path
from PIL import Image
import gradio as gr
from typing import List

# It imports the communication layer, not the API directly
from managers.gemini_manager import gemini_manager_singleton

logger = logging.getLogger(__name__)

class Deformes2DThinker:
    """
    The cognitive specialist that handles prompt engineering and creative logic.
    """
    def _read_prompt_template(self, filename: str) -> str:
        """Reads a prompt template file from the 'prompts' directory."""
        try:
            prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
            with open(prompts_dir / filename, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise gr.Error(f"Prompt template file not found: prompts/{filename}")

    def generate_storyboard(self, prompt: str, num_keyframes: int, ref_image_paths: List[str]) -> List[str]:
        """Acts as a Scriptwriter to generate a storyboard."""
        try:
            template = self._read_prompt_template("unified_storyboard_prompt.txt")
            storyboard_prompt = template.format(user_prompt=prompt, num_fragments=num_keyframes)
            images = [Image.open(p) for p in ref_image_paths]
            
            # Assemble all parts into a single list for the manager
            prompt_parts = [storyboard_prompt] + images
            storyboard_data = gemini_manager_singleton.get_json_object(prompt_parts)
            
            storyboard = storyboard_data.get("scene_storyboard", [])
            if not storyboard or len(storyboard) != num_keyframes: 
                raise ValueError(f"Incorrect number of scenes generated. Expected {num_keyframes}, got {len(storyboard)}.")
            return storyboard
        except Exception as e:
            raise gr.Error(f"The Scriptwriter (Deformes2D Thinker) failed: {e}")

    def select_keyframes_from_pool(self, storyboard: list, base_image_paths: list[str], pool_image_paths: list[str]) -> list[str]:
        """Acts as a Photographer/Editor to select keyframes."""
        if not pool_image_paths:
            raise gr.Error("The 'image pool' (Additional Images) is empty.")
            
        try:
            template = self._read_prompt_template("keyframe_selection_prompt.txt")
            
            image_map = {f"IMG-{i+1}": path for i, path in enumerate(pool_image_paths)}
            
            prompt_parts = ["# Reference Images (Story Base)"]
            prompt_parts.extend([Image.open(p) for p in base_image_paths])
            prompt_parts.append("\n# Image Pool (Scene Bank)")
            prompt_parts.extend([Image.open(p) for p in pool_image_paths])

            storyboard_str = "\n".join([f"- Scene {i+1}: {s}" for i, s in enumerate(storyboard)])
            selection_prompt = template.format(storyboard_str=storyboard_str, image_identifiers=list(image_map.keys()))
            prompt_parts.append(selection_prompt)

            selection_data = gemini_manager_singleton.get_json_object(prompt_parts)
            
            selected_identifiers = selection_data.get("selected_image_identifiers", [])
            
            if len(selected_identifiers) != len(storyboard):
                raise ValueError("The AI did not select the correct number of images for the scenes.")
            
            selected_paths = [image_map[identifier] for identifier in selected_identifiers]
            return selected_paths

        except Exception as e:
            raise gr.Error(f"The Photographer (Deformes2D Thinker) failed to select images: {e}")

    def get_anticipatory_keyframe_prompt(self, global_prompt: str, scene_history: str, current_scene_desc: str, future_scene_desc: str, last_image_path: str, fixed_ref_paths: list[str]) -> str:
        """Acts as an Art Director to generate an image prompt."""
        try:
            template = self._read_prompt_template("anticipatory_keyframe_prompt.txt")
            
            director_prompt = template.format(
                historico_prompt=scene_history,
                cena_atual=current_scene_desc,
                cena_futura=future_scene_desc
            )
            
            prompt_parts = [
                f"# CONTEXT:\n- Global Story Goal: {global_prompt}\n# VISUAL ASSETS:",
                "Current Base Image [IMG-BASE]:", 
                Image.open(last_image_path)
            ]
            
            ref_counter = 1
            for path in fixed_ref_paths:
                if path != last_image_path:
                    prompt_parts.extend([f"General Reference Image [IMG-REF-{ref_counter}]:", Image.open(path)])
                    ref_counter += 1

            prompt_parts.append(director_prompt)

            final_flux_prompt = gemini_manager_singleton.get_raw_text(prompt_parts)
            
            return final_flux_prompt.strip().replace("`", "").replace("\"", "")
        except Exception as e:
            raise gr.Error(f"The Art Director (Deformes2D Thinker) failed: {e}")

    def get_cinematic_decision(self, global_prompt: str, story_history: str, 
                               past_keyframe_path: str, present_keyframe_path: str, future_keyframe_path: str,
                               past_scene_desc: str, present_scene_desc: str, future_scene_desc: str) -> dict:
        """Acts as a Film Director to make editing decisions and generate motion prompts."""
        try:
            template = self._read_prompt_template("cinematic_director_prompt.txt")
            prompt_text = template.format(
                global_prompt=global_prompt, 
                story_history=story_history,
                past_scene_desc=past_scene_desc,
                present_scene_desc=present_scene_desc,
                future_scene_desc=future_scene_desc
            )
            
            prompt_parts = [
                prompt_text,
                "[PAST_IMAGE]:", Image.open(past_keyframe_path),
                "[PRESENT_IMAGE]:", Image.open(present_keyframe_path),
                "[FUTURE_IMAGE]:", Image.open(future_keyframe_path)
            ]
            
            decision_data = gemini_manager_singleton.get_json_object(prompt_parts)

            if "transition_type" not in decision_data or "motion_prompt" not in decision_data:
                raise ValueError("AI response (Cinematographer) is malformed. Missing 'transition_type' or 'motion_prompt'.")
            return decision_data
        except Exception as e:
            logger.error(f"The Film Director (Deformes2D Thinker) failed: {e}. Using fallback to 'continuous'.", exc_info=True)
            return {
                "transition_type": "continuous",
                "motion_prompt": f"A smooth, continuous cinematic transition from '{present_scene_desc}' to '{future_scene_desc}'."
            }

# --- Singleton Instance ---
deformes2d_thinker_singleton = Deformes2DThinker()