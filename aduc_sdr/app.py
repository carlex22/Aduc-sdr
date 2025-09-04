# app.py
#
# Copyright (C) August 4, 2025  Carlos Rodrigues dos Santos
#
# Version: 2.0.2
#
# Contact:
# Carlos Rodrigues dos Santos
# carlex22@gmail.com
#
# Related Repositories and Projects:
# GitHub: https://github.com/carlex22/Aduc-sdr
# YouTube (Results): https://m.youtube.com/channel/UC3EgoJi_Fv7yuDpvfYNtoIQ
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# PENDING PATENT NOTICE: The ADUC method and system implemented in this
# software is in the process of being patented. Please see NOTICE.md for details.

"""
This file serves as the main entry point for the ADUC-SDR Gradio user interface.
It orchestrates the multi-step workflow for AI-driven film creation, from
pre-production (storyboarding, keyframing) to production (original video rendering)
and post-production (upscaling, HD mastering, audio generation).

The UI is structured using Accordion blocks to guide the user through a logical
sequence of operations, while `gr.State` components manage the flow of data
(file paths of generated artifacts) between these independent steps.
"""

import gradio as gr
import yaml
import logging
import os
import sys
import shutil
import time
import json

from aduc_orchestrator import AducOrchestrator

# --- 1. CONFIGURATION AND INITIALIZATION ---
# This section sets up logging, loads internationalization strings, and initializes
# the core AducOrchestrator which manages all AI specialist models.

LOG_FILE_PATH = "aduc_log.txt"
if os.path.exists(LOG_FILE_PATH):
    os.remove(LOG_FILE_PATH)

log_format = '%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s] - %(message)s'
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter(log_format))
root_logger.addHandler(stream_handler)

file_handler = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(log_format))
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

# Load translation strings for the UI
i18n = {}
try:
    with open("i18n.json", "r", encoding="utf-8") as f:
        i18n = json.load(f)
except Exception as e:
    logger.error(f"Error loading i18n.json: {e}")
    i18n = {"pt": {}, "en": {}, "zh": {}}

# Fallback for missing languages
if 'pt' not in i18n: i18n['pt'] = i18n.get('en', {})
if 'en' not in i18n: i18n['en'] = {}
if 'zh' not in i18n: i18n['zh'] = i18n.get('en', {})

# Initialize the main orchestrator from the configuration file
try:
    with open("config.yaml", 'r') as f: config = yaml.safe_load(f)
    WORKSPACE_DIR = config['application']['workspace_dir']
    aduc = AducOrchestrator(workspace_dir=WORKSPACE_DIR)
    logger.info("ADUC Orchestrator and Specialists initialized successfully.")
except Exception as e:
    logger.error(f"CRITICAL ERROR during initialization: {e}", exc_info=True)
    exit()

# --- 2. UI WRAPPER FUNCTIONS ---
# These functions act as intermediaries between the Gradio UI components and the
# AducOrchestrator. They handle input validation, progress tracking, and updating
# the UI state after each operation.

def run_pre_production_wrapper(prompt, num_keyframes, ref_files, resolution_str, duration_per_fragment, progress=gr.Progress()):
    if not ref_files:
        raise gr.Error("Please provide at least one reference image.")

    ref_paths = [aduc.process_image_for_story(f.name, 480, f"ref_processed_{i}.png") for i, f in enumerate(ref_files)]

    progress(0.1, desc="Generating storyboard...")
    storyboard, initial_ref_path, _ = aduc.task_generate_storyboard(prompt, num_keyframes, ref_paths, progress)

    resolution = int(resolution_str.split('x')[0])

    def cb_factory(scene_index, total_scenes):
        start_time = time.time()
        total_steps = 12
        def callback(pipe_self, step, timestep, callback_kwargs):
            elapsed = time.time() - start_time
            current_step = step + 1
            if current_step > 0:
                it_per_sec = current_step / elapsed
                eta = (total_steps - current_step) / it_per_sec if it_per_sec > 0 else 0
                desc = f"Keyframe {scene_index}/{total_scenes}: {int((current_step/total_steps)*100)}% | {current_step}/{total_steps} [{elapsed:.0f}s<{eta:.0f}s, {it_per_sec:.2f}it/s]"
                base_progress = 0.2 + (scene_index - 1) * (0.8 / total_scenes)
                step_progress = (current_step / total_steps) * (0.8 / total_scenes)
                progress(base_progress + step_progress, desc=desc)
            return {}
        return callback

    final_keyframes = aduc.task_generate_keyframes(storyboard, initial_ref_path, prompt, resolution, cb_factory)

    return gr.update(value=storyboard), gr.update(value=final_keyframes), gr.update(visible=True, open=True)

def run_pre_production_photo_wrapper(prompt, num_keyframes, ref_files, progress=gr.Progress()):
    if not ref_files or len(ref_files) < 2:
        raise gr.Error("Photographer Mode requires at least 2 images: one base and one for the scene pool.")

    base_ref_paths = [aduc.process_image_for_story(ref_files[0].name, 480, "base_ref_processed_0.png")]
    pool_ref_paths = [aduc.process_image_for_story(f.name, 480, f"pool_ref_{i+1}.png") for i, f in enumerate(ref_files[1:])]

    progress(0.1, desc="Generating storyboard...")
    storyboard, _, _ = aduc.task_generate_storyboard(prompt, num_keyframes, base_ref_paths, progress)

    progress(0.5, desc="AI Photographer is selecting the best scenes...")
    selected_keyframes = aduc.task_select_keyframes(storyboard, base_ref_paths, pool_ref_paths)

    return gr.update(value=storyboard), gr.update(value=selected_keyframes), gr.update(visible=True, open=True)

def run_original_production_wrapper(keyframes, prompt, duration,
                                     trim_percent, handler_strength, destination_convergence_strength,
                                     guidance_scale, stg_scale, inference_steps,
                                     video_resolution,
                                     progress=gr.Progress()):
    """Wrapper for Step 3: Production. Correctly handles the return dictionary."""
    yield {
        original_video_output: gr.update(value=None, visible=True, label="🎬 Producing your original master video... Please wait."),
        final_video_output: gr.update(value=None, visible=True, label="🎬 Production in progress..."),
        step4_accordion: gr.update(visible=False)
    }

    resolution = int(video_resolution.split('x')[0])
    
    result = aduc.task_produce_original_movie(
        keyframes, prompt, duration,
        int(trim_percent), handler_strength, destination_convergence_strength,
        guidance_scale, stg_scale, int(inference_steps),
        resolution, use_continuity_director=True, progress=progress
    )
    
    original_latents = result["latent_paths"]
    original_video = result["final_path"]

    yield {
        original_video_output: gr.update(value=original_video, label="✅ Original Master Video"),
        final_video_output: gr.update(value=original_video, label="Final Film (Result of the Last Step)"),
        step4_accordion: gr.update(visible=True, open=True),
        original_latents_paths_state: original_latents,
        original_video_path_state: original_video,
        current_source_video_state: original_video,
    }

def run_upscaler_wrapper(latent_paths, chunk_size, progress=gr.Progress()):
    """Wrapper for Step 4A: Latent Upscaler. Correctly handles the generator."""
    if not latent_paths:
        raise gr.Error("Cannot run Upscaler. No original latents found. Please complete Step 3 first.")
        
    yield {
        upscaler_video_output: gr.update(value=None, visible=True, label="Upscaling latents and decoding video..."),
        final_video_output: gr.update(label="Post-Production in progress: Latent Upscaling...")
    }

    final_path = None
    for update in aduc.task_run_latent_upscaler(latent_paths, int(chunk_size), progress=progress):
        final_path = update['final_path']

    yield {
        upscaler_video_output: gr.update(value=final_path, label="✅ Latent Upscale Complete"),
        final_video_output: gr.update(value=final_path),
        upscaled_video_path_state: final_path,
        current_source_video_state: final_path,
    }

def run_hd_wrapper(source_video, model_version, steps, global_prompt, progress=gr.Progress()):
    """Wrapper for Step 4B: HD Mastering. Correctly handles the generator."""
    if not source_video:
        raise gr.Error("Cannot run HD Mastering. No source video found. Please complete a previous step first.")

    yield {
        hd_video_output: gr.update(value=None, visible=True, label="Applying HD mastering... This may take a while."),
        final_video_output: gr.update(label="Post-Production in progress: HD Mastering...")
    }
    
    final_path = None
    for update in aduc.task_run_hd_mastering(source_video, model_version, int(steps), global_prompt, progress=progress):
        final_path = update['final_path']

    yield {
        hd_video_output: gr.update(value=final_path, label="✅ HD Mastering Complete"),
        final_video_output: gr.update(value=final_path),
        hd_video_path_state: final_path,
        current_source_video_state: final_path,
    }

def run_audio_wrapper(source_video, audio_prompt, global_prompt, progress=gr.Progress()):
    """Wrapper for Step 4C: Audio Generation. Correctly handles the generator."""
    if not source_video:
        raise gr.Error("Cannot run Audio Generation. No source video found. Please complete a previous step first.")

    yield {
        audio_video_output: gr.update(value=None, visible=True, label="Generating audio and muxing..."),
        final_video_output: gr.update(label="Post-Production in progress: Audio Generation...")
    }

    final_audio_prompt = audio_prompt if audio_prompt and audio_prompt.strip() else global_prompt

    final_path = None
    for update in aduc.task_run_audio_generation(source_video, final_audio_prompt, progress=progress):
        final_path = update['final_path']

    yield {
        audio_video_output: gr.update(value=final_path, label="✅ Audio Generation Complete"),
        final_video_output: gr.update(value=final_path),
    }

def get_log_content():
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Log file not yet created. Start a generation."

def update_ui_language(lang_emoji):
    lang_code_map = {"🇧🇷": "pt", "🇺🇸": "en", "🇨🇳": "zh"}
    lang_code = lang_code_map.get(lang_emoji, "en")
    lang_map = i18n.get(lang_code, i18n.get('en', {}))
    return {
        # General
        title_md: gr.update(value=f"# {lang_map.get('app_title')}"),
        subtitle_md: gr.update(value=lang_map.get('app_subtitle')),
        lang_selector: gr.update(label=lang_map.get('lang_selector_label')),
        # Step 1: Pre-Production
        step1_accordion: gr.update(label=lang_map.get('step1_accordion')),
        prompt_input: gr.update(label=lang_map.get('prompt_label')),
        ref_image_input: gr.update(label=lang_map.get('ref_images_label')),
        num_keyframes_slider: gr.update(label=lang_map.get('keyframes_label')),
        duration_per_fragment_slider: gr.update(label=lang_map.get('duration_label'), info=lang_map.get('duration_info')),
        storyboard_and_keyframes_button: gr.update(value=lang_map.get('storyboard_and_keyframes_button')),
        storyboard_from_photos_button: gr.update(value=lang_map.get('storyboard_from_photos_button')),
        step1_mode_b_info_md: gr.update(value=f"*{lang_map.get('step1_mode_b_info')}*"),
        storyboard_output: gr.update(label=lang_map.get('storyboard_output_label')),
        keyframe_gallery: gr.update(label=lang_map.get('keyframes_gallery_label')),
        # Step 3: Production
        step3_accordion: gr.update(label=lang_map.get('step3_accordion')),
        step3_description_md: gr.update(value=lang_map.get('step3_description')),
        produce_original_button: gr.update(value=lang_map.get('produce_original_button')),
        ltx_advanced_options_accordion: gr.update(label=lang_map.get('ltx_advanced_options')),
        causality_accordion: gr.update(label=lang_map.get('causality_controls_title')),
        trim_percent_slider: gr.update(label=lang_map.get('trim_percent_label'), info=lang_map.get('trim_percent_info')),
        forca_guia_slider: gr.update(label=lang_map.get('forca_guia_label'), info=lang_map.get('forca_guia_info')),
        convergencia_destino_slider: gr.update(label=lang_map.get('convergencia_final_label'), info=lang_map.get('convergencia_final_info')),
        ltx_pipeline_accordion: gr.update(label=lang_map.get('ltx_pipeline_options')),
        guidance_scale_slider: gr.update(label=lang_map.get('guidance_scale_label'), info=lang_map.get('guidance_scale_info')),
        stg_scale_slider: gr.update(label=lang_map.get('stg_scale_label'), info=lang_map.get('stg_scale_info')),
        inference_steps_slider: gr.update(label=lang_map.get('steps_label'), info=lang_map.get('steps_info')),
        # Step 4: Post-Production
        step4_accordion: gr.update(label=lang_map.get('step4_accordion')),
        step4_description_md: gr.update(value=lang_map.get('step4_description')),
        sub_step_a_accordion: gr.update(label=lang_map.get('sub_step_a_upscaler')),
        upscaler_description_md: gr.update(value=lang_map.get('upscaler_description')),
        upscaler_options_accordion: gr.update(label=lang_map.get('upscaler_options')),
        upscaler_chunk_size_slider: gr.update(label=lang_map.get('upscaler_chunk_size_label'), info=lang_map.get('upscaler_chunk_size_info')),
        run_upscaler_button: gr.update(value=lang_map.get('run_upscaler_button')),
        sub_step_b_accordion: gr.update(label=lang_map.get('sub_step_b_hd')),
        hd_description_md: gr.update(value=lang_map.get('hd_description')),
        hd_options_accordion: gr.update(label=lang_map.get('hd_options')),
        hd_model_radio: gr.update(label=lang_map.get('hd_model_label')),
        hd_steps_slider: gr.update(label=lang_map.get('hd_steps_label'), info=lang_map.get('hd_steps_info')),
        run_hd_button: gr.update(value=lang_map.get('run_hd_button')),
        sub_step_c_accordion: gr.update(label=lang_map.get('sub_step_c_audio')),
        audio_description_md: gr.update(value=lang_map.get('audio_description')),
        audio_options_accordion: gr.update(label=lang_map.get('audio_options')),
        audio_prompt_input: gr.update(label=lang_map.get('audio_prompt_label'), info=lang_map.get('audio_prompt_info')),
        run_audio_button: gr.update(value=lang_map.get('run_audio_button')),
        # Final Outputs & Logs
        final_video_output: gr.update(label=lang_map.get('final_video_label')),
        log_accordion: gr.update(label=lang_map.get('log_accordion_label')),
        log_display: gr.update(label=lang_map.get('log_display_label')),
        update_log_button: gr.update(value=lang_map.get('update_log_button')),
    }

# --- 3. GRADIO UI DEFINITION ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    default_lang = i18n.get('pt', {})
    
    original_latents_paths_state = gr.State(value=None)
    original_video_path_state = gr.State(value=None)
    upscaled_video_path_state = gr.State(value=None)
    hd_video_path_state = gr.State(value=None)
    current_source_video_state = gr.State(value=None)

    title_md = gr.Markdown(f"# {default_lang.get('app_title')}")
    subtitle_md = gr.Markdown(default_lang.get('app_subtitle'))
    with gr.Row():
        lang_selector = gr.Radio(["🇧🇷", "🇺🇸", "🇨🇳"], value="🇧🇷", label=default_lang.get('lang_selector_label'))
        resolution_selector = gr.Radio(["480x480", "720x720", "960x960"], value="480x480", label="Base Resolution")

    with gr.Accordion(default_lang.get('step1_accordion'), open=True) as step1_accordion:
        prompt_input = gr.Textbox(label=default_lang.get('prompt_label'), value="A majestic lion walks across the savanna, sits down, and then roars at the setting sun.")
        ref_image_input = gr.File(label=default_lang.get('ref_images_label'), file_count="multiple", file_types=["image"])
        with gr.Row():
            num_keyframes_slider = gr.Slider(minimum=3, maximum=42, value=5, step=1, label=default_lang.get('keyframes_label'))
            duration_per_fragment_slider = gr.Slider(label=default_lang.get('duration_label'), info=default_lang.get('duration_info'), minimum=2.0, maximum=10.0, value=4.0, step=0.1)
        with gr.Row():
            storyboard_and_keyframes_button = gr.Button(default_lang.get('storyboard_and_keyframes_button'), variant="primary")
            storyboard_from_photos_button = gr.Button(default_lang.get('storyboard_from_photos_button'))
        step1_mode_b_info_md = gr.Markdown(f"*{default_lang.get('step1_mode_b_info')}*")
        storyboard_output = gr.JSON(label=default_lang.get('storyboard_output_label'))
        keyframe_gallery = gr.Gallery(label=default_lang.get('keyframes_gallery_label'), visible=True, object_fit="contain", height="auto", type="filepath")

    with gr.Accordion(default_lang.get('step3_accordion'), open=False, visible=False) as step3_accordion:
        step3_description_md = gr.Markdown(default_lang.get('step3_description'))
        with gr.Accordion(default_lang.get('ltx_advanced_options'), open=False) as ltx_advanced_options_accordion:
            with gr.Accordion(default_lang.get('causality_controls_title'), open=True) as causality_accordion:
                trim_percent_slider = gr.Slider(minimum=10, maximum=90, value=50, step=5, label=default_lang.get('trim_percent_label'), info=default_lang.get('trim_percent_info'))
                with gr.Row():
                    forca_guia_slider = gr.Slider(label=default_lang.get('forca_guia_label'), minimum=0.0, maximum=1.0, value=0.5, step=0.05, info=default_lang.get('forca_guia_info'))
                    convergencia_destino_slider = gr.Slider(label=default_lang.get('convergencia_final_label'), minimum=0.0, maximum=1.0, value=0.75, step=0.05, info=default_lang.get('convergencia_final_info'))
            with gr.Accordion(default_lang.get('ltx_pipeline_options'), open=True) as ltx_pipeline_accordion:
                with gr.Row():
                    guidance_scale_slider = gr.Slider(minimum=1.0, maximum=10.0, value=2.0, step=0.1, label=default_lang.get('guidance_scale_label'), info=default_lang.get('guidance_scale_info'))
                    stg_scale_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.025, step=0.005, label=default_lang.get('stg_scale_label'), info=default_lang.get('stg_scale_info'))
                inference_steps_slider = gr.Slider(minimum=10, maximum=50, value=20, step=1, label=default_lang.get('steps_label'), info=default_lang.get('steps_info'))
        produce_original_button = gr.Button(default_lang.get('produce_original_button'), variant="primary")
        original_video_output = gr.Video(label="Original Master Video", visible=False, interactive=False)

    with gr.Accordion(default_lang.get('step4_accordion'), open=False, visible=False) as step4_accordion:
        step4_description_md = gr.Markdown(default_lang.get('step4_description'))
        with gr.Accordion(default_lang.get('sub_step_a_upscaler'), open=True) as sub_step_a_accordion:
            upscaler_description_md = gr.Markdown(default_lang.get('upscaler_description'))
            with gr.Accordion(default_lang.get('upscaler_options'), open=False) as upscaler_options_accordion:
                upscaler_chunk_size_slider = gr.Slider(minimum=1, maximum=10, value=2, step=1, label=default_lang.get('upscaler_chunk_size_label'), info=default_lang.get('upscaler_chunk_size_info'))
            run_upscaler_button = gr.Button(default_lang.get('run_upscaler_button'), variant="secondary")
            upscaler_video_output = gr.Video(label="Upscaled Video", visible=False, interactive=False)
        with gr.Accordion(default_lang.get('sub_step_b_hd'), open=True) as sub_step_b_accordion:
            hd_description_md = gr.Markdown(default_lang.get('hd_description'))
            with gr.Accordion(default_lang.get('hd_options'), open=False) as hd_options_accordion:
                hd_model_radio = gr.Radio(["3B", "7B"], value="7B", label=default_lang.get('hd_model_label'))
                hd_steps_slider = gr.Slider(minimum=20, maximum=150, value=100, step=5, label=default_lang.get('hd_steps_label'), info=default_lang.get('hd_steps_info'))
            run_hd_button = gr.Button(default_lang.get('run_hd_button'), variant="secondary")
            hd_video_output = gr.Video(label="HD Mastered Video", visible=False, interactive=False)
        with gr.Accordion(default_lang.get('sub_step_c_audio'), open=True) as sub_step_c_accordion:
            audio_description_md = gr.Markdown(default_lang.get('audio_description'))
            with gr.Accordion(default_lang.get('audio_options'), open=False) as audio_options_accordion:
                audio_prompt_input = gr.Textbox(label=default_lang.get('audio_prompt_label'), info=default_lang.get('audio_prompt_info'), lines=3)
            run_audio_button = gr.Button(default_lang.get('run_audio_button'), variant="secondary")
            audio_video_output = gr.Video(label="Video with Audio", visible=False, interactive=False)

    final_video_output = gr.Video(label=default_lang.get('final_video_label'), visible=False, interactive=False)
    with gr.Accordion(default_lang.get('log_accordion_label'), open=False) as log_accordion:
        log_display = gr.Textbox(label=default_lang.get('log_display_label'), lines=20, interactive=False, autoscroll=True)
        update_log_button = gr.Button(default_lang.get('update_log_button'))

    # --- 4. UI EVENT CONNECTIONS ---
    all_ui_components = list(update_ui_language('🇧🇷').keys())
    lang_selector.change(fn=update_ui_language, inputs=lang_selector, outputs=all_ui_components)

    storyboard_and_keyframes_button.click(fn=run_pre_production_wrapper, inputs=[prompt_input, num_keyframes_slider, ref_image_input, resolution_selector, duration_per_fragment_slider], outputs=[storyboard_output, keyframe_gallery, step3_accordion])
    storyboard_from_photos_button.click(fn=run_pre_production_photo_wrapper, inputs=[prompt_input, num_keyframes_slider, ref_image_input], outputs=[storyboard_output, keyframe_gallery, step3_accordion])

    produce_original_button.click(
        fn=run_original_production_wrapper,
        inputs=[keyframe_gallery, prompt_input, duration_per_fragment_slider, trim_percent_slider, forca_guia_slider, convergencia_destino_slider, guidance_scale_slider, stg_scale_slider, inference_steps_slider, resolution_selector],
        outputs=[original_video_output, final_video_output, step4_accordion, original_latents_paths_state, original_video_path_state, current_source_video_state]
    )
    
    run_upscaler_button.click(
        fn=run_upscaler_wrapper,
        inputs=[original_latents_paths_state, upscaler_chunk_size_slider],
        outputs=[upscaler_video_output, final_video_output, upscaled_video_path_state, current_source_video_state]
    )
    
    run_hd_button.click(
        fn=run_hd_wrapper,
        inputs=[current_source_video_state, hd_model_radio, hd_steps_slider, prompt_input],
        outputs=[hd_video_output, final_video_output, hd_video_path_state, current_source_video_state]
    )

    run_audio_button.click(
        fn=run_audio_wrapper,
        inputs=[current_source_video_state, audio_prompt_input, prompt_input],
        outputs=[audio_video_output, final_video_output]
    )
    
    update_log_button.click(fn=get_log_content, inputs=[], outputs=[log_display])

# --- 5. APPLICATION LAUNCH ---
if __name__ == "__main__":
    if os.path.exists(WORKSPACE_DIR):
        logger.info(f"Clearing previous workspace at: {WORKSPACE_DIR}")
        shutil.rmtree(WORKSPACE_DIR)
    os.makedirs(WORKSPACE_DIR)
    logger.info(f"Application started. Launching Gradio interface...")
    demo.queue().launch()