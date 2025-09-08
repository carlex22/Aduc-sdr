# aduc_orchestrator.py
#
# Copyright (C) August 4, 2025  Carlos Rodrigues dos Santos
#
# Version: 2.2.0
#
# This file contains the core ADUC (Automated Discovery and Orchestration of Complex tasks)
# orchestrator, known as the "Maestro" (Γ). Its responsibility is to manage the high-level
# creative workflow of film production. This version is updated to reflect the final
# refactored project structure with `engineers` and `managers`.

import os
import logging
from typing import List, Dict, Any, Generator, Tuple

import gradio as gr
from PIL import Image, ImageOps

from engineers.deformes4D import Deformes4DEngine
from engineers.deformes2D_thinker import deformes2d_thinker_singleton
from engineers.deformes3D import deformes3d_engine_singleton

# The logger is configured in app.py; here we just get the instance.
logger = logging.getLogger(__name__)

class AducDirector:
    """
    Represents the Scene Director, responsible for managing the production state.
    Acts as the "score" for the orchestra, keeping track of all generated artifacts
    (script, keyframes, etc.) during the creative process.
    """
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)
        self.state: Dict[str, Any] = {}
        logger.info(f"The stage is set. Workspace at '{self.workspace_dir}'.")

    def update_state(self, key: str, value: Any) -> None:
        logger.info(f"Notating on the score: State '{key}' updated.")
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

class AducOrchestrator:
    """
    Implements the Maestro (Γ), the central orchestration layer of the ADUC architecture.
    It does not execute AI tasks directly but delegates each step of the creative
    process (scriptwriting, art direction, cinematography) to the appropriate Specialists.
    """
    def __init__(self, workspace_dir: str):
        self.director = AducDirector(workspace_dir)
        self.editor = Deformes4DEngine(workspace_dir)
        self.painter = deformes3d_engine_singleton
        logger.info("ADUC Maestro is on the podium. Musicians (specialists) are ready.")

    def process_image_for_story(self, image_path: str, size: int, filename: str) -> str:
        """
        Pre-processes a reference image, standardizing it for use by the Specialists.
        """
        img = Image.open(image_path).convert("RGB")
        img_square = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
        processed_path = os.path.join(self.director.workspace_dir, filename)
        img_square.save(processed_path)
        logger.info(f"Reference image processed and saved to: {processed_path}")
        return processed_path

    # --- PRE-PRODUCTION TASKS ---

    def task_generate_storyboard(self, prompt: str, num_keyframes: int, ref_image_paths: List[str],
                                 progress: gr.Progress) -> Tuple[List[str], str, Any]:
        """
        Delegates the task of creating the storyboard to the Scriptwriter (deformes2D_thinker).
        """
        logger.info(f"Act 1, Scene 1: Script. Instructing Scriptwriter to create {num_keyframes} scenes.")
        progress(0.2, desc="Consulting AI Scriptwriter...")

        storyboard = deformes2d_thinker_singleton.generate_storyboard(prompt, num_keyframes, ref_image_paths)

        logger.info(f"Scriptwriter returned the score: {storyboard}")
        self.director.update_state("storyboard", storyboard)
        self.director.update_state("processed_ref_paths", ref_image_paths)
        return storyboard, ref_image_paths[0], gr.update(visible=True, open=True)

    def task_select_keyframes(self, storyboard: List[str], base_ref_paths: List[str],
                              pool_ref_paths: List[str]) -> List[str]:
        """
        Delegates to the Photographer (deformes2D_thinker) the task of selecting keyframes.
        """
        logger.info(f"Act 1, Scene 2 (Photographer Mode): Instructing Photographer to select {len(storyboard)} keyframes.")
        selected_paths = deformes2d_thinker_singleton.select_keyframes_from_pool(storyboard, base_ref_paths, pool_ref_paths)
        logger.info(f"Photographer selected the following scenes: {[os.path.basename(p) for p in selected_paths]}")
        self.director.update_state("keyframes", selected_paths)
        return selected_paths

    def task_generate_keyframes(self, storyboard: List[str], initial_ref_path: str, global_prompt: str,
                                keyframe_resolution: int, progress_callback_factory=None) -> List[str]:
        """
        Delegates to the Art Director (Deformes3DEngine) the task of generating keyframes.
        """
        logger.info("Act 1, Scene 2 (Art Director Mode): Delegating to Art Director.")
        general_ref_paths = self.director.get_state("processed_ref_paths", [])

        final_keyframes = self.painter.generate_keyframes_from_storyboard(
            storyboard=storyboard,
            initial_ref_path=initial_ref_path,
            global_prompt=global_prompt,
            keyframe_resolution=keyframe_resolution,
            general_ref_paths=general_ref_paths,
            progress_callback_factory=progress_callback_factory
        )
        self.director.update_state("keyframes", final_keyframes)
        logger.info("Maestro: Art Director has completed keyframe generation.")
        return final_keyframes

    # --- PRODUCTION & POST-PRODUCTION TASKS ---

    def task_produce_original_movie(self, keyframes: List[str], global_prompt: str, seconds_per_fragment: float,
                                    trim_percent: int, handler_strength: float,
                                    destination_convergence_strength: float,
                                    guidance_scale: float, stg_scale: float, inference_steps: int,
                                    video_resolution: int, use_continuity_director: bool,
                                    progress: gr.Progress) -> Dict[str, Any]:
        """
        Delegates the production of the original master video to the Deformes4DEngine.
        """
        logger.info("Maestro: Delegating production of the original movie to Deformes4DEngine.")
        storyboard = self.director.get_state("storyboard", [])

        result = self.editor.generate_original_movie(
            keyframes=keyframes,
            global_prompt=global_prompt,
            storyboard=storyboard,
            seconds_per_fragment=seconds_per_fragment,
            trim_percent=trim_percent,
            handler_strength=handler_strength,
            destination_convergence_strength=destination_convergence_strength,
            video_resolution=video_resolution,
            use_continuity_director=use_continuity_director,
            guidance_scale=guidance_scale,
            stg_scale=stg_scale,
            num_inference_steps=inference_steps,
            progress=progress
        )
        
        self.director.update_state("final_video_path", result["final_path"])
        self.director.update_state("latent_paths", result["latent_paths"])
        logger.info("Maestro: Original movie production complete.")
        return result

    def task_run_latent_upscaler(self, latent_paths: List[str], chunk_size: int, progress: gr.Progress) -> Generator[Dict[str, Any], None, None]:
        """
        Orchestrates the latent upscaling task.
        """
        logger.info(f"Maestro: Delegating latent upscaling task for {len(latent_paths)} fragments.")
        for update in self.editor.upscale_latents_and_create_video(
            latent_paths=latent_paths,
            chunk_size=chunk_size,
            progress=progress
        ):
            if "final_path" in update and update["final_path"]:
                self.director.update_state("final_video_path", update["final_path"])
                yield update
                break
        logger.info("Maestro: Latent upscaling complete.")
    
    def task_run_hd_mastering(self, source_video_path: str, model_version: str, steps: int, prompt: str, progress: gr.Progress) -> Generator[Dict[str, Any], None, None]:
        """
        Orchestrates the HD mastering task.
        """
        logger.info(f"Maestro: Delegating HD mastering task using SeedVR {model_version}.")
        for update in self.editor.master_video_hd(
            source_video_path=source_video_path,
            model_version=model_version,
            steps=steps,
            prompt=prompt,
            progress=progress
        ):
            if "final_path" in update and update["final_path"]:
                self.director.update_state("final_video_path", update["final_path"])
                yield update
                break
        logger.info("Maestro: HD mastering complete.")

    def task_run_audio_generation(self, source_video_path: str, audio_prompt: str, progress: gr.Progress) -> Generator[Dict[str, Any], None, None]:
        """
        Orchestrates the audio generation task.
        """
        logger.info(f"Maestro: Delegating audio generation task.")
        for update in self.editor.generate_audio_for_final_video(
            source_video_path=source_video_path,
            audio_prompt=audio_prompt,
            progress=progress
        ):
             if "final_path" in update and update["final_path"]:
                self.director.update_state("final_video_path", update["final_path"])
                yield update
                break
        logger.info("Maestro: Audio generation complete.")