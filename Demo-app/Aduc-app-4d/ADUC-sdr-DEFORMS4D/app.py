# app.py (Versão Final: Feedback visual com yield, sem JS)
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos

import gradio as gr
import yaml
import logging
import os
import shutil
import time
import json

from aduc_orchestrator import AducOrchestrator

# --- 1. CONFIGURAÇÃO E INICIALIZAÇÃO ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

i18n = {}
try:
    with open("i18n.json", "r", encoding="utf-8") as f:
        i18n = json.load(f)
except FileNotFoundError:
    logger.warning("Arquivo i18n.json não encontrado! A interface usará textos em inglês como fallback.")
except json.JSONDecodeError:
    logger.error("Erro ao decodificar i18n.json. Verifique a formatação do arquivo.")

if 'pt' not in i18n:
    i18n['pt'] = i18n.get('en', {})

try:
    with open("config.yaml", 'r') as f: config = yaml.safe_load(f)
    WORKSPACE_DIR = config['application']['workspace_dir']
    aduc = AducOrchestrator(workspace_dir=WORKSPACE_DIR)
    logger.info("Orquestrador ADUC e Especialistas inicializados com sucesso.")
except Exception as e:
    logger.error(f"ERRO CRÍTICO ao inicializar: {e}", exc_info=True)
    exit()

# --- 2. WRAPPERS DA UI ---

def preprocess_uploaded_images_wrapper(uploaded_files):
    if not uploaded_files: return None
    processed_paths = []
    base_resolution_for_analysis = 480
    for i, file_obj in enumerate(uploaded_files):
        processed_paths.append(aduc.editor.process_image_for_story(file_obj.name, base_resolution_for_analysis, f"user_ref_processed_{i}.png"))
    return processed_paths

def run_storyboard_wrapper(prompt, num_keyframes, processed_ref_image_paths, progress=gr.Progress()):
    if not processed_ref_image_paths: raise gr.Error("Por favor, forneça pelo menos uma imagem de referência.")
    return aduc.task_generate_storyboard(prompt, num_keyframes, processed_ref_image_paths, progress)

def run_keyframes_wrapper(storyboard, initial_ref_path, global_prompt, keyframe_resolution, use_art_director, progress=gr.Progress()):
    if not storyboard: raise gr.Error("Gere um roteiro primeiro (Etapa 1).")
    if not initial_ref_path: raise gr.Error("A imagem de referência inicial não foi processada.")
    
    resolution = int(keyframe_resolution.split('x')[0])

    def cb_factory(scene_index, total_scenes):
        start_time = time.time()
        total_steps = 24
        def callback(pipe_self, step, timestep, callback_kwargs):
            elapsed = time.time() - start_time
            current_step = step + 1
            if current_step > 0:
                it_per_sec = current_step / elapsed
                eta = (total_steps - current_step) / it_per_sec if it_per_sec > 0 else 0
                desc = f"Keyframe {scene_index}/{total_scenes}: {int((current_step/total_steps)*100)}% | {current_step}/{total_steps} [{elapsed:.0f}s<{eta:.0f}s, {it_per_sec:.2f}it/s]"
                progress(current_step / total_steps, desc=desc)
            return {}
        return callback
    final_keyframes = aduc.task_generate_keyframes(storyboard, initial_ref_path, global_prompt, resolution, use_art_director, cb_factory)
    return gr.update(value=final_keyframes), gr.update(visible=True, open=True)

def handle_manual_keyframes_wrapper(uploaded_files):
    if not uploaded_files or len(uploaded_files) < 2:
        raise gr.Error("Por favor, carregue pelo menos 2 imagens para servir de keyframes.")
    keyframe_paths = [file.name for file in uploaded_files]
    return gr.update(value=keyframe_paths), gr.update(visible=True, open=True)

def run_video_production_wrapper(keyframes, prompt, duration, n_corte, n_eco, p_caminho, p_dest, 
                                 guidance, stg, rescaling, video_resolution, use_cont, use_cine,
                                 progress=gr.Progress()):
    # PASSO 1: ATUALIZAÇÃO INICIAL. Limpa e mostra os componentes.
    yield {
        video_fragments_gallery: gr.update(value=None, visible=True),
        final_video_output: gr.update(value=None, visible=True, label="🎬 Produzindo seu filme... Por favor, aguarde.")
    }
    
    adv_params = {"guidance_scale": guidance, "stg_scale": stg, "rescaling_scale": rescaling}
    resolution = int(video_resolution.split('x')[0])

    video_fragments_so_far = []
    final_movie_path = None
    
    # PASSO 2: LOOP DE GERAÇÃO COM FEEDBACK
    for update in aduc.task_produce_final_movie_with_feedback(
        keyframes, prompt, duration, n_corte, n_eco, p_caminho, p_dest, 
        adv_params, resolution, use_cont, use_cine, progress
    ):
        if "fragment_path" in update and update["fragment_path"]:
            video_fragments_so_far.append(update["fragment_path"])
            yield {
                video_fragments_gallery: gr.update(value=video_fragments_so_far),
                final_video_output: gr.update() # Mantém o estado atual (carregando)
            }
        elif "final_path" in update and update["final_path"]:
            final_movie_path = update["final_path"]
            # O loop terminou, vamos para a atualização final.
            break
    
    # PASSO 3: ATUALIZAÇÃO FINAL
    yield {
        video_fragments_gallery: gr.update(), # Mantém a galeria como está
        final_video_output: gr.update(value=final_movie_path, label="🎉 FILME COMPLETO 🎉")
    }

def update_ui_language(lang_code):
    lang_map = i18n.get(lang_code, i18n.get('en', {}))
    return {
        title_md: gr.update(value=f"# {lang_map.get('app_title')}"),
        subtitle_md: gr.update(value=lang_map.get('app_subtitle')),
        lang_selector: gr.update(label=lang_map.get('lang_selector_label')),
        step1_accordion: gr.update(label=lang_map.get('step1_accordion')),
        prompt_input: gr.update(label=lang_map.get('prompt_label')),
        initial_ref_image_input: gr.update(label=lang_map.get('ref_images_label')),
        num_keyframes_slider: gr.update(label=lang_map.get('keyframes_label')),
        storyboard_button: gr.update(value=lang_map.get('storyboard_button')),
        storyboard_output: gr.update(label=lang_map.get('storyboard_output_label')),
        step2_accordion: gr.update(label=lang_map.get('step2_accordion')),
        step2_description_md: gr.update(value=lang_map.get('step2_description')),
        art_director_checkbox: gr.update(label=lang_map.get('art_director_label')),
        keyframe_button: gr.update(value=lang_map.get('keyframes_button')),
        manual_keyframe_upload: gr.update(label=lang_map.get('manual_keyframes_label')),
        manual_keyframe_separator: gr.update(value=lang_map.get('manual_separator')),
        keyframe_gallery: gr.update(label=lang_map.get('keyframes_gallery_label')),
        step3_accordion: gr.update(label=lang_map.get('step3_accordion')),
        step3_description_md: gr.update(value=lang_map.get('step3_description')),
        continuity_director_checkbox: gr.update(label=lang_map.get('continuity_director_label')),
        cinematographer_checkbox: gr.update(label=lang_map.get('cinematographer_label')),
        duration_per_fragment_slider: gr.update(label=lang_map.get('duration_label')),
        n_corte_slider: gr.update(label=lang_map.get('n_corte_label'), info=lang_map.get('n_corte_info')),
        eco_slider: gr.update(label=lang_map.get('eco_label')),
        dejavu_slider: gr.update(label=lang_map.get('path_label')),
        p_dest_slider: gr.update(label=lang_map.get('dest_label')),
        produce_button: gr.update(value=lang_map.get('produce_button')),
        advanced_accordion: gr.update(label=lang_map.get('advanced_accordion_label')),
        guidance_scale_slider: gr.update(label=lang_map.get('guidance_label')),
        stg_scale_slider: gr.update(label=lang_map.get('stg_label')),
        rescaling_scale_slider: gr.update(label=lang_map.get('rescaling_label')),
        video_fragments_gallery: gr.update(label=lang_map.get('video_fragments_gallery_label')),
        final_video_output: gr.update(label=lang_map.get('final_movie_with_audio_label')),
    }

# --- 4. DEFINIÇÃO DA UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    default_lang = i18n.get('pt', {})
    
    title_md = gr.Markdown(f"# {default_lang.get('app_title')}")
    subtitle_md = gr.Markdown(default_lang.get('app_subtitle'))
    
    with gr.Row():
        lang_selector = gr.Radio(["pt", "en", "zh"], value="pt", label=default_lang.get('lang_selector_label'))
        resolution_selector = gr.Radio(["480x480", "512x736", "736x1280"], value="480x480", label="Resolução do Vídeo")

    with gr.Accordion(default_lang.get('step1_accordion'), open=True) as step1_accordion:
        prompt_input = gr.Textbox(label=default_lang.get('prompt_label'), value="A majestic lion walks across the savanna, sits down, and then roars at the setting sun.")
        with gr.Row():
            initial_ref_image_input = gr.File(label=default_lang.get('ref_images_label'), file_count="multiple", file_types=["image"])
            num_keyframes_slider = gr.Slider(minimum=2, maximum=10, value=3, step=1, label=default_lang.get('keyframes_label'))
        storyboard_button = gr.Button(default_lang.get('storyboard_button'), variant="primary")
        storyboard_output = gr.JSON(label=default_lang.get('storyboard_output_label'))
        
    with gr.Accordion(default_lang.get('step2_accordion'), open=False, visible=False) as step2_accordion:
        step2_description_md = gr.Markdown(default_lang.get('step2_description'))
        art_director_checkbox = gr.Checkbox(label=default_lang.get('art_director_label'), value=True)
        keyframe_button = gr.Button(default_lang.get('keyframes_button'), variant="primary")
        manual_keyframe_separator = gr.Markdown(default_lang.get('manual_separator'))
        manual_keyframe_upload = gr.File(label=default_lang.get('manual_keyframes_label'), file_count="multiple", file_types=["image"])
        keyframe_gallery = gr.Gallery(label=default_lang.get('keyframes_gallery_label'), visible=True, object_fit="contain", height="auto")
        keyframe_ref_image_state = gr.State()

    with gr.Accordion(default_lang.get('step3_accordion'), open=False, visible=False) as step3_accordion:
        step3_description_md = gr.Markdown(default_lang.get('step3_description'))
        with gr.Row():
            continuity_director_checkbox = gr.Checkbox(label=default_lang.get('continuity_director_label'), value=True)
            cinematographer_checkbox = gr.Checkbox(label=default_lang.get('cinematographer_label'), value=True)
        with gr.Row():
            duration_per_fragment_slider = gr.Slider(label=default_lang.get('duration_label'), minimum=2.0, maximum=10.0, value=4.0, step=0.1)
            n_corte_slider = gr.Slider(label=default_lang.get('n_corte_label'), minimum=10, maximum=50, value=25, step=1, info=default_lang.get('n_corte_info'))
            eco_slider = gr.Slider(label=default_lang.get('eco_label'), minimum=8, maximum=32, value=8, step=8)
        with gr.Row():
            dejavu_slider = gr.Slider(label=default_lang.get('path_label'), minimum=0.0, maximum=1.0, value=0.5, step=0.05)
            p_dest_slider = gr.Slider(label=default_lang.get('dest_label'), minimum=0.0, maximum=1.0, value=1.0, step=0.05)
        with gr.Accordion(default_lang.get('advanced_accordion_label'), open=False) as advanced_accordion:
             with gr.Row():
                guidance_scale_slider = gr.Slider(label=default_lang.get('guidance_label'), minimum=1.0, maximum=15.0, value=8.0, step=0.5)
                stg_scale_slider = gr.Slider(label=default_lang.get('stg_label'), minimum=0.0, maximum=10.0, value=4.0, step=0.5)
                rescaling_scale_slider = gr.Slider(label=default_lang.get('rescaling_label'), minimum=0.0, maximum=1.0, value=0.5, step=0.05)
        produce_button = gr.Button(default_lang.get('produce_button'), variant="primary")
    
    video_fragments_gallery = gr.Gallery(label="Fragmentos do Filme", visible=False, object_fit="contain", height="auto", type="filepath")
    final_video_output = gr.Video(label="Filme Completo", visible=False)

    # --- 5. CONEXÕES DA UI ---
    
    ui_components_to_translate = [
        title_md, subtitle_md, lang_selector, step1_accordion, prompt_input, 
        initial_ref_image_input, num_keyframes_slider, storyboard_button, 
        storyboard_output, step2_accordion, step2_description_md, art_director_checkbox,
        keyframe_button, manual_keyframe_upload, manual_keyframe_separator, keyframe_gallery,
        step3_accordion, step3_description_md, 
        continuity_director_checkbox, cinematographer_checkbox,
        duration_per_fragment_slider, n_corte_slider, eco_slider, dejavu_slider, p_dest_slider, 
        produce_button, advanced_accordion, guidance_scale_slider, 
        stg_scale_slider, rescaling_scale_slider, video_fragments_gallery, final_video_output
    ]
    lang_selector.change(fn=update_ui_language, inputs=lang_selector, outputs=ui_components_to_translate)
    
    initial_ref_image_input.upload(fn=preprocess_uploaded_images_wrapper, inputs=[initial_ref_image_input], outputs=[initial_ref_image_input])
    
    storyboard_button.click(fn=run_storyboard_wrapper, inputs=[prompt_input, num_keyframes_slider, initial_ref_image_input], outputs=[storyboard_output, keyframe_ref_image_state, step2_accordion])
    
    keyframe_button.click(fn=run_keyframes_wrapper, inputs=[storyboard_output, keyframe_ref_image_state, prompt_input, resolution_selector, art_director_checkbox], outputs=[keyframe_gallery, step3_accordion])
    
    manual_keyframe_upload.upload(fn=handle_manual_keyframes_wrapper, inputs=[manual_keyframe_upload], outputs=[keyframe_gallery, step3_accordion])
    
    produce_button.click(
        fn=run_video_production_wrapper,
        inputs=[
            keyframe_gallery, prompt_input, duration_per_fragment_slider, n_corte_slider, eco_slider,
            dejavu_slider, p_dest_slider, guidance_scale_slider, stg_scale_slider, rescaling_scale_slider,
            resolution_selector, continuity_director_checkbox, cinematographer_checkbox
        ],
        outputs=[video_fragments_gallery, final_video_output]
    )

if __name__ == "__main__":
    if os.path.exists(WORKSPACE_DIR): shutil.rmtree(WORKSPACE_DIR)
    os.makedirs(WORKSPACE_DIR)
    demo.queue().launch()