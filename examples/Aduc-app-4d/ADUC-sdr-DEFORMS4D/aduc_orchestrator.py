# aduc_orchestrator.py (CORRIGIDO com função geradora para feedback visual)
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos

import os
import time
import shutil
import logging
import gradio as gr
from PIL import Image
import subprocess
from pathlib import Path

from deformes4D_engine import Deformes4DEngine
from ltx_manager_helpers import ltx_manager_singleton
from gemini_helpers import gemini_singleton

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quantize_to_multiple(n, m):
    if m == 0:
        return n
    else:
        return int(round(n / m) * m)

class AducDirector:
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)
        self.state = {}
        logger.info(f"Diretor ADUC inicializado. Workspace em '{self.workspace_dir}' está pronto.")

    def reset(self):
        os.makedirs(self.workspace_dir, exist_ok=True)
        self.state = {}
        logger.info("Estado do Diretor resetado.")

    def update_state(self, key, value):
        log_value = value if not isinstance(value, (dict, list)) and not hasattr(value, 'shape') else f"Objeto complexo"
        logger.info(f"Diretor: Atualizando estado '{key}' -> '{log_value}'")
        self.state[key] = value

    def get_state(self, key, default=None):
        return self.state.get(key, default)

class AducOrchestrator:
    def __init__(self, workspace_dir: str):
        self.director = AducDirector(workspace_dir)
        self.editor = Deformes4DEngine(ltx_manager_singleton, workspace_dir)

    def task_generate_storyboard(self, prompt, num_keyframes, processed_ref_image_paths, progress):
        progress(0.4, desc="Consultando Roteirista IA (Gemini)...")
        storyboard = gemini_singleton.generate_storyboard(
            prompt, num_keyframes, processed_ref_image_paths
        )
        self.director.update_state("storyboard", storyboard)
        self.director.update_state("processed_ref_paths", processed_ref_image_paths)
        return storyboard, processed_ref_image_paths[0], gr.update(visible=True, open=True)

    def task_generate_keyframes(self, storyboard, initial_ref_path, global_prompt, keyframe_resolution, use_art_director, progress_callback_factory=None):
        all_fixed_refs = self.director.get_state("processed_ref_paths", [])
        keyframes_for_movie = [] 
        current_ref_path = initial_ref_path
        story_history = "The story begins with the main character."
        width, height = int(keyframe_resolution), int(keyframe_resolution)
        
        logger.info(f"Iniciando geração de {len(storyboard)} keyframes com resolução {width}x{height}.")
        for i, scene_desc in enumerate(storyboard):
            progress_callback = progress_callback_factory(i + 1, len(storyboard)) if progress_callback_factory else None
            
            if use_art_director:
                keyframe_prompt, selected_ref_paths = gemini_singleton.get_keyframe_prompt(
                    global_prompt=global_prompt, scene_history=story_history,
                    current_scene_desc=scene_desc, last_image_path=current_ref_path,
                    fixed_ref_paths=all_fixed_refs
                )
            else:
                keyframe_prompt = scene_desc
                selected_ref_paths = list(set([current_ref_path] + all_fixed_refs))

            selected_ref_images_pil = [Image.open(p) for p in selected_ref_paths]
            
            new_keyframe_path = self.editor.generate_keyframe(
                prompt=keyframe_prompt, reference_images=selected_ref_images_pil,
                output_filename=f"keyframe_{i+1}.png", width=width, height=height,
                callback=progress_callback
            )
            
            keyframes_for_movie.append(new_keyframe_path)
            current_ref_path = new_keyframe_path
            story_history += f"\n- Scene {i+1}: {scene_desc}"
            logger.info(f"Keyframe {i+1} gerado: {os.path.basename(new_keyframe_path)}")
            
        self.director.update_state("keyframes", keyframes_for_movie)
        return keyframes_for_movie
    
    def task_produce_final_movie_with_feedback(self, keyframes, global_prompt, duration_per_fragment, 
                           n_corte_percent, n_eco, p_caminho, p_dest,
                           ltx_advanced_params,
                           video_resolution, use_continuity_director, 
                           use_cinematographer, progress):
        if not keyframes or len(keyframes) < 2:
            raise gr.Error("Pelo menos 2 keyframes são necessários para produzir um filme.")
        
        keyframe_paths = [item[0] if isinstance(item, tuple) else item for item in keyframes]
        storyboard = self.director.get_state("storyboard", [])
        
        video_clips_with_audio = []
        latent_fragment_paths = []
        
        current_latent_path = None
        story_history = ""
        num_transitions = len(keyframe_paths) - 1
        
        target_resolution_tuple = (int(video_resolution), int(video_resolution))
        a_frames_per_scene = quantize_to_multiple(int(duration_per_fragment * 24), 8) + 1
        n_corte_frames = quantize_to_multiple(int(duration_per_fragment * 24 * (n_corte_percent / 100.0)), 8)
        n_eco_frames = n_eco

        for i in range(num_transitions):
            progress((i + 1) / (num_transitions + 1), desc=f"Cena {i+1}/{num_transitions}")
            
            start_media_path = keyframe_paths[i]
            end_media_path = keyframe_paths[i+1]
            scene_prompt = storyboard[i] if i < len(storyboard) else "A cena final."
            ltx_params = {"p_dest": p_dest, "p_caminho": p_caminho, **ltx_advanced_params}
            new_latent = None
            new_video = None

            if current_latent_path is None:
                if use_cinematographer:
                    motion_prompt = gemini_singleton.get_initial_motion_prompt(global_prompt, start_media_path, end_media_path, scene_prompt)
                else:
                    motion_prompt = scene_prompt
                ltx_params["motion_prompt"] = motion_prompt
                
                new_latent, new_video = self.editor.create_initial_fragment(
                    start_media_path, end_media_path, duration_per_fragment, ltx_params, 
                    target_resolution_tuple, scene_prompt
                )
                current_latent_path = new_latent
                transition_type = "start"
            else:
                if use_continuity_director and use_cinematographer:
                    memory_image = self.editor.get_last_frame_from_latent(current_latent_path)
                    path_scene_desc = storyboard[i-1] if i > 0 else storyboard[0]
                    decision = gemini_singleton.get_transition_decision(global_prompt, story_history, memory_image, start_media_path, end_media_path, path_scene_desc, scene_prompt)
                    transition_type, motion_prompt = decision["transition_type"], decision["motion_prompt"]
                else:
                    transition_type, motion_prompt = "continuous", scene_prompt
                
                ltx_params["motion_prompt"] = motion_prompt
                
                if transition_type == "cut":
                    bridge_video = self.editor.create_ffmpeg_bridge(start_media_path, end_media_path, 0.5, 24, target_resolution_tuple)
                    video_clips_with_audio.append(bridge_video)
                    yield {"fragment_path": bridge_video} # Envia a ponte para a UI
                    current_latent_path = None
                    continue

                new_latent, new_video = self.editor.create_next_fragment(
                    current_latent_path, end_media_path, n_corte_frames, n_eco_frames, 
                    a_frames_per_scene, p_caminho, ltx_params, target_resolution_tuple, scene_prompt
                )
                current_latent_path = new_latent

            if new_latent and new_video:
                latent_fragment_paths.append(new_latent)
                video_clips_with_audio.append(new_video)
                yield {"fragment_path": new_video} # Envia o clipe pronto para a UI
            
            story_history += f"\n- Ato {i + 1} ({transition_type}): {motion_prompt}"

        progress(1.0, desc="Montagem final do filme...")
        final_movie_path = os.path.join(self.director.workspace_dir, f"final_movie_{int(time.time())}.mp4")
        
        self.editor.concatenate_videos_ffmpeg(video_clips_with_audio, final_movie_path)
        self.director.update_state("final_video_path", final_movie_path)
        
        logger.info("Concatenando latentes individuais para o arquivo final...")
        self.editor.concatenate_latents_only(latent_fragment_paths, n_eco_frames)
            
        yield {"final_path": final_movie_path}