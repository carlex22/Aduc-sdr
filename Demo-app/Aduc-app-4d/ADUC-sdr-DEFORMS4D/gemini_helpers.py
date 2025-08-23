# gemini_helpers.py (Final - Com todos os papéis do Diretor IA, incluindo Áudio)
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos

import os
import logging
import json
import gradio as gr
from PIL import Image
import google.generativeai as genai
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def robust_json_parser(raw_text: str) -> dict:
    clean_text = raw_text.strip()
    try:
        start_index = clean_text.find('{')
        end_index = clean_text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = clean_text[start_index : end_index + 1]
            return json.loads(json_str)
        else: raise ValueError("Nenhum objeto JSON válido encontrado na resposta da IA.")
    except json.JSONDecodeError as e:
        logger.error(f"Falha ao decodificar JSON. String recebida:\n{raw_text}")
        raise ValueError(f"Falha ao decodificar JSON: {e}")

class GeminiSingleton:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-pro')
            logger.info("Especialista Gemini (1.5 Flash) inicializado com sucesso.")
        else:
            self.model = None
            logger.warning("Chave da API Gemini não encontrada. Especialista desativado.")
            
    def _check_model(self):
        if not self.model:
            raise gr.Error("A chave da API do Google Gemini não está configurada (GEMINI_API_KEY).")

    def _read_prompt_template(self, filename: str) -> str:
        try:
            with open(os.path.join("prompts", filename), "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            raise gr.Error(f"Arquivo de prompt não encontrado: prompts/{filename}")

    def generate_storyboard(self, prompt: str, num_keyframes: int, ref_image_paths: list[str]) -> list[str]:
        self._check_model()
        try:
            template = self._read_prompt_template("unified_storyboard_prompt.txt")
            storyboard_prompt = template.format(user_prompt=prompt, num_fragments=num_keyframes, image_metadata="N/A")
            model_contents = [storyboard_prompt] + [Image.open(p) for p in ref_image_paths]
            response = self.model.generate_content(model_contents)
            storyboard_data = robust_json_parser(response.text)
            storyboard = storyboard_data.get("scene_storyboard", [])
            if not storyboard or len(storyboard) != num_keyframes: raise ValueError(f"Número incorreto de cenas gerado.")
            return storyboard
        except Exception as e:
            raise gr.Error(f"O Roteirista (Gemini) falhou: {e}")

    def get_keyframe_prompt(self, global_prompt: str, scene_history: str, current_scene_desc: str, last_image_path: str, fixed_ref_paths: list[str]) -> tuple[str, list[str]]:
        self._check_model()
        try:
            template = self._read_prompt_template("director_composition_prompt.txt")
            director_prompt = template.format(global_prompt=global_prompt, scene_history=scene_history, current_scene_desc=current_scene_desc)
            image_map = {1: last_image_path}
            model_contents = [f"Last Generated Image [IMG-1]:", Image.open(last_image_path)]
            current_image_index = 2
            for path in fixed_ref_paths:
                if path not in image_map.values():
                    image_map[current_image_index] = path
                    model_contents.extend([f"Fixed Reference Image [IMG-{current_image_index}]:", Image.open(path)])
                    current_image_index += 1
            model_contents.append(director_prompt)
            response = self.model.generate_content(model_contents)
            composition_prompt_with_tags = response.text.strip()
            referenced_tags = re.findall(r'\[IMG-(\d+)\]', composition_prompt_with_tags)
            selected_image_paths = [image_map[int(tag)] for tag in referenced_tags if int(tag) in image_map]
            if not selected_image_paths: selected_image_paths.append(last_image_path)
            final_kontext_prompt = re.sub(r'\[IMG-\d+\]', '', composition_prompt_with_tags).strip()
            wrapper_template = self._read_prompt_template("flux_composition_wrapper_prompt.txt")
            return wrapper_template.format(target_prompt=final_kontext_prompt), list(set(selected_image_paths))
        except Exception as e:
            raise gr.Error(f"O Diretor de Arte (Gemini) falhou: {e}")

    def get_initial_motion_prompt(self, user_prompt: str, start_image_path: str, destination_image_path: str, dest_scene_desc: str) -> str:
        self._check_model()
        try:
            template = self._read_prompt_template("initial_motion_prompt.txt")
            prompt_text = template.format(user_prompt=user_prompt, destination_scene_description=dest_scene_desc)
            model_contents = [prompt_text, "START Image:", Image.open(start_image_path), "DESTINATION Image:", Image.open(destination_image_path)]
            response = self.model.generate_content(model_contents)
            return response.text.strip()
        except Exception as e:
            raise gr.Error(f"O Cineasta Inicial (Gemini) falhou: {e}")

    def get_transition_decision(self, user_prompt: str, story_history: str, memory_image: Image.Image, path_image_path: str, destination_image_path: str, midpoint_scene_description: str, destination_scene_description: str) -> dict:
        self._check_model()
        try:
            template = self._read_prompt_template("transition_decision_prompt.txt")
            prompt_text = template.format(
                user_prompt=user_prompt, story_history=story_history, 
                midpoint_scene_description=midpoint_scene_description, 
                destination_scene_description=destination_scene_description
            )
            model_contents = ["START Image (from last generated frame):", memory_image, "MIDPOINT Image (Path):", Image.open(path_image_path), "DESTINATION Image (Destination):", Image.open(destination_image_path), prompt_text]
            response = self.model.generate_content(model_contents)
            decision_data = robust_json_parser(response.text)
            if "transition_type" not in decision_data or "motion_prompt" not in decision_data: raise ValueError("Resposta da IA está mal formatada.")
            return decision_data
        except Exception as e:
            raise gr.Error(f"O Diretor de Continuidade (Gemini) falhou: {e}")
            
    def generate_audio_prompts(self, global_prompt: str, storyboard: list) -> dict:
        self._check_model()
        try:
            template = self._read_prompt_template("audio_director_prompt.txt")
            storyboard_str = "\n".join([f"- {s}" for s in storyboard])
            prompt_text = template.format(global_prompt=global_prompt, storyboard_str=storyboard_str)
            response = self.model.generate_content(prompt_text)
            audio_prompts = robust_json_parser(response.text)
            if "music_prompt" not in audio_prompts or "sfx_prompt" not in audio_prompts:
                raise ValueError("Resposta da IA para áudio está mal formatada.")
            return audio_prompts
        except Exception as e:
            logger.error(f"O Diretor de Áudio (Gemini) falhou: {e}. Usando prompts padrão.")
            return {
                "music_prompt": "dramatic cinematic score",
                "sfx_prompt": "natural environmental sounds"
            }

gemini_singleton = GeminiSingleton()