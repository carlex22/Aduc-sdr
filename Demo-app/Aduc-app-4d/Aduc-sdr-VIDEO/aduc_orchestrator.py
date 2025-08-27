# aduc_orchestrator.py
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos
#
# Este programa é software livre: você pode redistribuí-lo e/ou modificá-lo
# sob os termos da Licença Pública Geral Affero GNU...
# AVISO DE PATENTE PENDENTE: Consulte NOTICE.md.

import os
import time
import shutil
import logging
import gradio as gr
from PIL import Image, ImageOps
import subprocess
from pathlib import Path
import json

from deformes4D_engine import Deformes4DEngine
from ltx_manager_helpers import ltx_manager_singleton
from gemini_helpers import gemini_singleton
from image_specialist import image_specialist_singleton

# Configuração de logging centralizada deve ser feita no app.py
logger = logging.getLogger(__name__)

class AducDirector:
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)
        self.state = {}
        logger.info(f"O palco está pronto. Workspace em '{self.workspace_dir}'.")

    def reset(self):
        os.makedirs(self.workspace_dir, exist_ok=True)
        self.state = {}
        logger.info("Partitura limpa. Estado do Diretor reiniciado.")

    def update_state(self, key, value):
        log_value = value if not isinstance(value, (dict, list)) and not hasattr(value, 'shape') else f"Objeto complexo"
        logger.info(f"Anotando na partitura: Estado '{key}' atualizado.")
        self.state[key] = value

    def get_state(self, key, default=None):
        return self.state.get(key, default)

class AducOrchestrator:
    def __init__(self, workspace_dir: str):
        self.director = AducDirector(workspace_dir)
        self.editor = Deformes4DEngine(ltx_manager_singleton, workspace_dir)
        self.painter = image_specialist_singleton
        logger.info("Maestro ADUC está no pódio. Músicos (especialistas) prontos.")

    def process_image_for_story(self, image_path: str, size: int, filename: str = None) -> str:
        """
        Pré-processa uma imagem de referência: converte para RGB, redimensiona para um
        quadrado e salva no diretório de trabalho.
        """
        img = Image.open(image_path).convert("RGB")
        img_square = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
        
        if filename: 
            processed_path = os.path.join(self.director.workspace_dir, filename)
        else: 
            processed_path = os.path.join(self.director.workspace_dir, f"ref_processed_{int(time.time()*1000)}.png")
            
        img_square.save(processed_path)
        logger.info(f"Imagem de referência processada e salva em: {processed_path}")
        return processed_path

    def task_generate_storyboard(self, prompt, num_keyframes, processed_ref_image_paths, progress):
        logger.info(f"Ato 1, Cena 1: Roteiro. Instruindo o Roteirista (Gemini) a criar {num_keyframes} cenas a partir de: '{prompt}'")
        progress(0.2, desc="Consultando Roteirista IA (Gemini)...")
        storyboard = gemini_singleton.generate_storyboard(prompt, num_keyframes, processed_ref_image_paths)
        logger.info(f"Roteirista retornou a partitura: {storyboard}")
        self.director.update_state("storyboard", storyboard)
        self.director.update_state("processed_ref_paths", processed_ref_image_paths)
        return storyboard, processed_ref_image_paths[0], gr.update(visible=True, open=True)

    def task_select_keyframes(self, storyboard, base_ref_paths, pool_ref_paths):
        logger.info(f"Ato 1, Cena 2 (Alternativa): Fotografia. Instruindo o Editor (Gemini) a selecionar {len(storyboard)} keyframes de um banco de {len(pool_ref_paths)} imagens.")
        selected_paths = gemini_singleton.select_keyframes_from_pool(storyboard, base_ref_paths, pool_ref_paths)
        logger.info(f"Editor selecionou as seguintes cenas: {[os.path.basename(p) for p in selected_paths]}")
        self.director.update_state("keyframes", selected_paths)
        return selected_paths

    def task_generate_keyframes(self, storyboard, initial_ref_path, global_prompt, keyframe_resolution, progress_callback_factory=None):
        """
        Delega a tarefa de geração de keyframes para o ImageSpecialist.
        """
        logger.info(f"Ato 1, Cena 2: Direção de Arte. Delegando ao Especialista de Imagem.")
        
        general_ref_paths = self.director.get_state("processed_ref_paths", [])
        
        final_keyframes = self.painter.generate_keyframes_from_storyboard(
            storyboard=storyboard,
            initial_ref_path=initial_ref_path,
            global_prompt=global_prompt,
            keyframe_resolution=int(keyframe_resolution),
            general_ref_paths=general_ref_paths,
            progress_callback_factory=progress_callback_factory
        )
        
        self.director.update_state("keyframes", final_keyframes)
        logger.info("Maestro: Especialista de Imagem concluiu a geração dos keyframes.")
        return final_keyframes
    
    def task_produce_final_movie_with_feedback(self, keyframes, global_prompt, seconds_per_fragment, 
                           overlap_percent, echo_frames,
                           handler_strength, 
                           destination_convergence_strength,
                           base_ltx_params,
                           video_resolution, use_continuity_director, 
                           use_cinematographer, progress): 
        
        logger.info("AducOrchestrator: Delegando a produção do filme completo ao Deformes4DEngine.")
        storyboard = self.director.get_state("storyboard", [])

        for update in self.editor.generate_full_movie(
            keyframes=keyframes, 
            global_prompt=global_prompt, 
            storyboard=storyboard, 
            seconds_per_fragment=seconds_per_fragment, 
            overlap_percent=overlap_percent, 
            echo_frames=echo_frames, 
            handler_strength=handler_strength, 
            destination_convergence_strength=destination_convergence_strength,
            base_ltx_params=base_ltx_params,
            video_resolution=video_resolution, 
            use_continuity_director=use_continuity_director, 
            progress=progress
        ):
            if "fragment_path" in update and update["fragment_path"]:
                yield {"fragment_path": update["fragment_path"]}
            elif "final_path" in update and update["final_path"]:
                final_movie_path = update["final_path"]
                self.director.update_state("final_video_path", final_movie_path)
                yield {"final_path": final_movie_path}
                break

        logger.info("AducOrchestrator: Produção do filme concluída e estado do diretor atualizado.")