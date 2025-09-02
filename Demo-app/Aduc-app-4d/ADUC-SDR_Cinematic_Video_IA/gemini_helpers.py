#Uma implementação aberta e funcional da arquitetura ADUC-SDR para geração de vídeo coerente.
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos
#
# Contato:
# Carlos Rodrigues dos Santos
# carlex22@gmail.com
#
# Repositórios e Projetos Relacionados:
# GitHub: https://github.com/carlex22/Aduc-sdr
# YouTube (Resultados): https://m.youtube.com/channel/UC3EgoJi_Fv7yuDpvfYNtoIQ
#
# Este programa é software livre: você pode redistribuí-lo e/ou modificá-lo
# sob os termos da Licença Pública Geral Affero da GNU como publicada pela
# Free Software Foundation, seja a versão 3 da Licença, ou
# (a seu critério) qualquer versão posterior.
#
# Este programa é distribuído na esperança de que seja útil,
# mas SEM QUALQUER GARANTIA; sem mesmo a garantia implícita de
# COMERCIALIZAÇÃO ou ADEQUAÇÃO A UM DETERMINADO FIM. Consulte a
# Licença Pública Geral Affero da GNU para mais detalhes.
#
# Você deve ter recebido uma cópia da Licença Pública Geral Affero da GNU
# junto com este programa. Se não, veja <https://www.gnu.org/licenses/>.
#
# AVISO DE PATENTE PENDENTE: O método e sistema ADUC implementado neste 
# software está em processo de patenteamento. Consulte NOTICE.md.

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
        # Tenta encontrar o JSON delimitado por ```json ... ```
        match = re.search(r'```json\s*(\{.*?\})\s*```', clean_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        
        # Se não encontrar, tenta encontrar o primeiro '{' e o último '}'
        start_index = clean_text.find('{')
        end_index = clean_text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = clean_text[start_index : end_index + 1]
            return json.loads(json_str)
        else:
            raise ValueError("Nenhum objeto JSON válido foi encontrado na resposta da IA.")
    except json.JSONDecodeError as e:
        logger.error(f"Falha ao decodificar JSON. A IA retornou o seguinte texto:\n---\n{raw_text}\n---")
        raise ValueError(f"A IA retornou um formato de JSON inválido: {e}")

class GeminiSingleton:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            # Modelo mais recente e capaz para tarefas complexas de visão e raciocínio.
            self.model = genai.GenerativeModel('gemini-2.0-flash') 
            logger.info("Especialista Gemini (1.5 Pro) inicializado com sucesso.")
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
            storyboard_prompt = template.format(user_prompt=prompt, num_fragments=num_keyframes)
            model_contents = [storyboard_prompt] + [Image.open(p) for p in ref_image_paths]
            response = self.model.generate_content(model_contents)
            
            logger.info(f"--- RESPOSTA COMPLETA DO GEMINI (generate_storyboard) ---\n{response.text}\n--------------------")
            
            storyboard_data = robust_json_parser(response.text)
            storyboard = storyboard_data.get("scene_storyboard", [])
            if not storyboard or len(storyboard) != num_keyframes: raise ValueError(f"Número incorreto de cenas gerado.")
            return storyboard
        except Exception as e:
            raise gr.Error(f"O Roteirista (Gemini) falhou: {e}")

    def select_keyframes_from_pool(self, storyboard: list, base_image_paths: list[str], pool_image_paths: list[str]) -> list[str]:
        self._check_model()
        if not pool_image_paths:
            raise gr.Error("O 'banco de imagens' (Imagens Adicionais) está vazio.")
            
        try:
            template = self._read_prompt_template("keyframe_selection_prompt.txt")
            
            image_map = {f"IMG-{i+1}": path for i, path in enumerate(pool_image_paths)}
            base_image_map = {f"BASE-{i+1}": path for i, path in enumerate(base_image_paths)}
            
            model_contents = ["# Reference Images (Story Base)"]
            for identifier, path in base_image_map.items():
                model_contents.extend([f"Identifier: {identifier}", Image.open(path)])
            
            model_contents.append("\n# Image Pool (Scene Bank)")
            for identifier, path in image_map.items():
                model_contents.extend([f"Identifier: {identifier}", Image.open(path)])

            storyboard_str = "\n".join([f"- Scene {i+1}: {s}" for i, s in enumerate(storyboard)])
            selection_prompt = template.format(storyboard_str=storyboard_str, image_identifiers=list(image_map.keys()))
            model_contents.append(selection_prompt)
            
            response = self.model.generate_content(model_contents)
            
            logger.info(f"--- RESPOSTA COMPLETA DO GEMINI (select_keyframes_from_pool) ---\n{response.text}\n--------------------")
            
            selection_data = robust_json_parser(response.text)
            selected_identifiers = selection_data.get("selected_image_identifiers", [])
            
            if len(selected_identifiers) != len(storyboard):
                raise ValueError("A IA não selecionou o número correto de imagens para as cenas.")
            
            selected_paths = [image_map[identifier] for identifier in selected_identifiers]
            return selected_paths

        except Exception as e:
            raise gr.Error(f"O Fotógrafo (Gemini) falhou ao selecionar as imagens: {e}")

    def get_anticipatory_keyframe_prompt(self, global_prompt: str, scene_history: str, current_scene_desc: str, future_scene_desc: str, last_image_path: str, fixed_ref_paths: list[str]) -> str:
        self._check_model()
        try:
            template = self._read_prompt_template("anticipatory_keyframe_prompt.txt")
            
            director_prompt = template.format(
                historico_prompt=scene_history,
                cena_atual=current_scene_desc,
                cena_futura=future_scene_desc
            )
            
            model_contents = [
                "# CONTEXTO:",
                f"- Global Story Goal: {global_prompt}",
                "# VISUAL ASSETS:",
                "Current Base Image [IMG-BASE]:", 
                Image.open(last_image_path)
            ]
            
            ref_counter = 1
            for path in fixed_ref_paths:
                if path != last_image_path:
                    model_contents.extend([f"General Reference Image [IMG-REF-{ref_counter}]:", Image.open(path)])
                    ref_counter += 1

            model_contents.append(director_prompt)

            response = self.model.generate_content(model_contents)
            
            logger.info(f"--- RESPOSTA COMPLETA DO GEMINI (get_anticipatory_keyframe_prompt) ---\n{response.text}\n--------------------")

            final_flux_prompt = response.text.strip()
            return final_flux_prompt
        except Exception as e:
            raise gr.Error(f"O Diretor de Arte (Gemini) falhou: {e}")

    def get_initial_motion_prompt(self, user_prompt: str, start_image_path: str, destination_image_path: str, dest_scene_desc: str) -> str:
        """Gera o prompt de movimento para a PRIMEIRA transição, que não tem um 'passado'."""
        self._check_model()
        try:
            template = self._read_prompt_template("initial_motion_prompt.txt")
            prompt_text = template.format(user_prompt=user_prompt, destination_scene_description=dest_scene_desc)
            model_contents = [
                prompt_text, 
                "START Image:", 
                Image.open(start_image_path), 
                "DESTINATION Image:", 
                Image.open(destination_image_path)
            ]
            response = self.model.generate_content(model_contents)
            
            logger.info(f"--- RESPOSTA COMPLETA DO GEMINI (get_initial_motion_prompt) ---\n{response.text}\n--------------------")

            return response.text.strip()
        except Exception as e:
            raise gr.Error(f"O Cineasta Inicial (Gemini) falhou: {e}")

    def get_cinematic_decision(self, global_prompt: str, story_history: str, 
                               past_keyframe_path: str, present_keyframe_path: str, future_keyframe_path: str,
                               past_scene_desc: str, present_scene_desc: str, future_scene_desc: str) -> dict:
        """
        Atua como um 'Cineasta', analisando passado, presente e futuro para tomar decisões
        de edição e gerar prompts de movimento detalhados.
        """
        self._check_model()
        try:
            template = self._read_prompt_template("cinematic_director_prompt.txt")
            prompt_text = template.format(
                global_prompt=global_prompt, 
                story_history=story_history,
                past_scene_desc=past_scene_desc,
                present_scene_desc=present_scene_desc,
                future_scene_desc=future_scene_desc
            )
            
            model_contents = [
                prompt_text,
                "[PAST_IMAGE]:", Image.open(past_keyframe_path),
                "[PRESENT_IMAGE]:", Image.open(present_keyframe_path),
                "[FUTURE_IMAGE]:", Image.open(future_keyframe_path)
            ]
            
            response = self.model.generate_content(model_contents)
            
            logger.info(f"--- RESPOSTA COMPLETA DO GEMINI (get_cinematic_decision) ---\n{response.text}\n--------------------")
            
            decision_data = robust_json_parser(response.text)
            if "transition_type" not in decision_data or "motion_prompt" not in decision_data:
                raise ValueError("Resposta da IA (Cineasta) está mal formatada. Faltam 'transition_type' ou 'motion_prompt'.")
            return decision_data
        except Exception as e:
            # Fallback para uma decisão segura em caso de erro
            logger.error(f"O Diretor de Cinema (Gemini) falhou: {e}. Usando fallback para 'continuous'.")
            return {
                "transition_type": "continuous",
                "motion_prompt": f"A smooth, continuous cinematic transition from '{present_scene_desc}' to '{future_scene_desc}'."
            }
            
            
          
    def get_sound_director_prompt(self, audio_history: str, 
                                  past_keyframe_path: str, present_keyframe_path: str, future_keyframe_path: str,
                                  present_scene_desc: str, motion_prompt: str, future_scene_desc: str) -> str:
        """
        Atua como um 'Diretor de Som', analisando o contexto completo para criar um prompt
        de áudio imersivo e contínuo para a cena atual.
        """
        self._check_model()
        try:
            template = self._read_prompt_template("sound_director_prompt.txt")
            prompt_text = template.format(
                audio_history=audio_history,
                present_scene_desc=present_scene_desc,
                motion_prompt=motion_prompt,
                future_scene_desc=future_scene_desc
            )
            
            model_contents = [
                prompt_text,
                "[PAST_IMAGE]:", Image.open(past_keyframe_path),
                "[PRESENT_IMAGE]:", Image.open(present_keyframe_path),
                "[FUTURE_IMAGE]:", Image.open(future_keyframe_path)
            ]
            
            response = self.model.generate_content(model_contents)
            
            logger.info(f"--- RESPOSTA COMPLETA DO GEMINI (get_sound_director_prompt) ---\n{response.text}\n--------------------")
            
            return response.text.strip()
        except Exception as e:
            logger.error(f"O Diretor de Som (Gemini) falhou: {e}. Usando fallback.")
            return f"Sound effects matching the scene: {present_scene_desc}"
            

gemini_singleton = GeminiSingleton()