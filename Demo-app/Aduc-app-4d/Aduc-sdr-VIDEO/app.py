# app.py
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos
#
# Este programa é software livre: você pode redistribuí-lo e/ou modificá-lo
# sob os termos da Licença Pública Geral Affero GNU como publicada pela
# Free Software Foundation, seja a versão 3 da Licença, ou
# (a seu critério) qualquer versão posterior.
#
# AVISO DE PATENTE PENDENTE: O método e sistema ADUC implementado neste 
# software está em processo de patenteamento. Consulte NOTICE.md.

import gradio as gr
import yaml
import logging
import os
import sys
import shutil
import time
import json

from aduc_orchestrator import AducOrchestrator

# --- 1. CONFIGURAÇÃO E INICIALIZAÇÃO ---

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

i18n = {}
try:
    with open("i18n.json", "r", encoding="utf-8") as f:
        i18n = json.load(f)
except FileNotFoundError:
    logger.warning("Arquivo i18n.json não encontrado! A interface usará textos em inglês como fallback.")
except json.JSONDecodeError:
    logger.error("Erro ao decodificar i18n.json. Verifique a formatação do arquivo.")

if 'pt' not in i18n: i18n['pt'] = i18n.get('en', {})
if 'en' not in i18n: i18n['en'] = {}
if 'zh' not in i18n: i18n['zh'] = i18n.get('en', {})

try:
    with open("config.yaml", 'r') as f: config = yaml.safe_load(f)
    WORKSPACE_DIR = config['application']['workspace_dir']
    aduc = AducOrchestrator(workspace_dir=WORKSPACE_DIR)
    logger.info("Orquestrador ADUC e Especialistas inicializados com sucesso.")
except Exception as e:
    logger.error(f"ERRO CRÍTICO ao inicializar: {e}", exc_info=True)
    exit()

# --- 2. WRAPPERS DA UI ---

def preprocess_base_images_wrapper(uploaded_files):
    if not uploaded_files: return None
    processed_paths = [aduc.process_image_for_story(f.name, 480, f"ref_processed_{i}.png") for i, f in enumerate(uploaded_files)]
    return gr.update(value=processed_paths)

def run_mode_a_wrapper(prompt, num_keyframes, ref_files, resolution_str, duration_per_fragment, progress=gr.Progress()):
    if not ref_files: 
        raise gr.Error("Por favor, forneça pelo menos uma imagem de referência.")
    
    ref_paths = [f.name for f in ref_files]

    progress(0.1, desc="Gerando roteiro...")
    storyboard, initial_ref_path, _ = aduc.task_generate_storyboard(prompt, num_keyframes, ref_paths, progress)
    
    resolution = int(resolution_str.split('x')[0])

    def cb_factory(scene_index, total_scenes):
        start_time = time.time()
        total_steps = 30
        def callback(pipe_self, step, timestep, callback_kwargs):
            elapsed = time.time() - start_time
            current_step = step + 1
            if current_step > 0:
                it_per_sec = current_step / elapsed
                eta = (total_steps - current_step) / it_per_sec if it_per_sec > 0 else 0
                desc = f"Keyframe {scene_index}/{total_scenes}: {int((current_step/total_steps)*100)}% | {current_step}/{total_steps} [{elapsed:.0f}s<{eta:.0f}s, {it_per_sec:.2f}it/s]"
                progress(0.2 + (current_step / total_steps) * 0.8, desc=desc)
            return {}
        return callback
    
    final_keyframes = aduc.task_generate_keyframes(storyboard, initial_ref_path, prompt, resolution, cb_factory)
    
    return gr.update(value=storyboard), gr.update(value=final_keyframes), gr.update(visible=True, open=True)

def run_mode_b_wrapper(prompt, num_keyframes, ref_files, progress=gr.Progress()):
    if not ref_files or len(ref_files) < 2: 
        raise gr.Error("Modo Fotógrafo requer pelo menos 2 imagens: uma base e uma para o banco de cenas.")

    base_ref_paths = [aduc.process_image_for_story(ref_files[0].name, 480, "base_ref_processed_0.png")]
    pool_ref_paths = [f.name for f in ref_files[1:]]

    progress(0.1, desc="Gerando roteiro...")
    storyboard, _, _ = aduc.task_generate_storyboard(prompt, num_keyframes, base_ref_paths, progress)
    
    progress(0.5, desc="IA (Fotógrafo) está selecionando as melhores cenas...")
    selected_keyframes = aduc.task_select_keyframes(storyboard, base_ref_paths, pool_ref_paths)
    
    return gr.update(value=storyboard), gr.update(value=selected_keyframes), gr.update(visible=True, open=True)

def run_video_production_wrapper(keyframes, prompt, duration, overlap_percent, echo_frames, 
                                 handler_strength, destination_convergence_strength,
                                 guidance, stg, rescaling, num_inference_steps,
                                 video_resolution, use_cont, use_cine,
                                 progress=gr.Progress()):
    yield {
        video_fragments_gallery: gr.update(value=None, visible=True),
        final_video_output: gr.update(value=None, visible=True, label="🎬 Produzindo seu filme... Por favor, aguarde.")
    }
    
    adv_params = {
        "guidance_scale": guidance, "stg_scale": stg, "rescaling_scale": rescaling,
        "num_inference_steps": num_inference_steps
    }
    resolution = int(video_resolution.split('x')[0])

    video_fragments_so_far = []
    final_movie_path = None
    
    for update in aduc.task_produce_final_movie_with_feedback(
        keyframes, prompt, duration, overlap_percent, echo_frames, 
        handler_strength, destination_convergence_strength,
        adv_params, resolution, use_cont, use_cine, progress
    ):
        if "fragment_path" in update and update["fragment_path"]:
            video_fragments_so_far.append(update["fragment_path"])
            yield { video_fragments_gallery: gr.update(value=video_fragments_so_far), final_video_output: gr.update() }
        elif "final_path" in update and update["final_path"]:
            final_movie_path = update["final_path"]
            break
    
    yield {
        video_fragments_gallery: gr.update(),
        final_video_output: gr.update(value=final_movie_path, label="🎉 FILME COMPLETO 🎉")
    }

def get_log_content():
    """Função para ler e retornar o conteúdo do arquivo de log."""
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Arquivo de log ainda não criado. Inicie uma geração."

def update_ui_language(lang_code):
    lang_map = i18n.get(lang_code, i18n.get('en', {}))
    # ... (a função de tradução permanece a mesma, mas está aqui para completude)
    return {
        title_md: gr.update(value=f"# {lang_map.get('app_title')}"),
        subtitle_md: gr.update(value=lang_map.get('app_subtitle')),
        lang_selector: gr.update(label=lang_map.get('lang_selector_label')),
        step1_accordion: gr.update(label=lang_map.get('step1_accordion')),
        prompt_input: gr.update(label=lang_map.get('prompt_label')),
        ref_image_input: gr.update(label=lang_map.get('ref_images_label')),
        num_keyframes_slider: gr.update(label=lang_map.get('keyframes_label')),
        duration_per_fragment_slider: gr.update(label=lang_map.get('duration_label')),
        storyboard_and_keyframes_button: gr.update(value=lang_map.get('storyboard_and_keyframes_button')),
        storyboard_from_photos_button: gr.update(value=lang_map.get('storyboard_from_photos_button')),
        storyboard_output: gr.update(label=lang_map.get('storyboard_output_label')),
        keyframe_gallery: gr.update(label=lang_map.get('keyframes_gallery_label')),
        step3_accordion: gr.update(label=lang_map.get('step3_accordion')),
        step3_description_md: gr.update(value=lang_map.get('step3_description')),
        continuity_director_checkbox: gr.update(label=lang_map.get('continuity_director_label')),
        cinematographer_checkbox: gr.update(label=lang_map.get('cinematographer_label')),
        echo_frames_selector: gr.update(label=lang_map.get('echo_frames_label'), info=lang_map.get('echo_frames_info')),
        overlap_percent_slider: gr.update(label=lang_map.get('overlap_percent_label'), info=lang_map.get('overlap_percent_info')),
        handler_strength_slider: gr.update(label=lang_map.get('handler_strength_label'), info=lang_map.get('handler_strength_info')),
        destination_convergence_slider: gr.update(label=lang_map.get('destination_convergence_label'), info=lang_map.get('destination_convergence_info')),
        produce_button: gr.update(value=lang_map.get('produce_button')),
        advanced_accordion: gr.update(label=lang_map.get('advanced_accordion_label')),
        guidance_scale_slider: gr.update(label=lang_map.get('guidance_label')),
        stg_scale_slider: gr.update(label=lang_map.get('stg_label')),
        rescaling_scale_slider: gr.update(label=lang_map.get('rescaling_label')),
        num_inference_steps_slider: gr.update(label=lang_map.get('steps_label'), info=lang_map.get('steps_info')),
        video_fragments_gallery: gr.update(label=lang_map.get('video_fragments_gallery_label')),
        final_video_output: gr.update(label=lang_map.get('final_movie_with_audio_label')),
        log_accordion: gr.update(label=lang_map.get('log_accordion_label')),
        log_display: gr.update(label=lang_map.get('log_display_label')),
        update_log_button: gr.update(value=lang_map.get('update_log_button')),
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
            num_keyframes_slider = gr.Slider(minimum=3, maximum=10, value=3, step=1, label=default_lang.get('keyframes_label'), info="Mínimo de 3 para a lógica do cineasta.")
            duration_per_fragment_slider = gr.Slider(label=default_lang.get('duration_label'), minimum=2.0, maximum=10.0, value=4.0, step=0.1)
        ref_image_input = gr.File(label=default_lang.get('ref_images_label'), file_count="multiple", file_types=["image"])
        with gr.Row():
            storyboard_and_keyframes_button = gr.Button(default_lang.get('storyboard_and_keyframes_button'), variant="primary")
            storyboard_from_photos_button = gr.Button(default_lang.get('storyboard_from_photos_button'))
        gr.Markdown(f"*{default_lang.get('step1_mode_b_info')}*")
        storyboard_output = gr.JSON(label=default_lang.get('storyboard_output_label'))
        keyframe_gallery = gr.Gallery(label=default_lang.get('keyframes_gallery_label'), visible=True, object_fit="contain", height="auto", type="filepath")
        
    with gr.Accordion(default_lang.get('step3_accordion'), open=False, visible=False) as step3_accordion:
        step3_description_md = gr.Markdown(default_lang.get('step3_description'))
        with gr.Row():
            continuity_director_checkbox = gr.Checkbox(label=default_lang.get('continuity_director_label'), value=True)
            cinematographer_checkbox = gr.Checkbox(label=default_lang.get('cinematographer_label'), value=True, visible=False)
        
        gr.Markdown("--- \n**Controles de Continuidade e Edição:**")
        with gr.Row():
            echo_frames_selector = gr.Radio(choices=[8, 16, 24], value=8, label=default_lang.get('echo_frames_label'), info=default_lang.get('echo_frames_info'))
            overlap_percent_slider = gr.Slider(label=default_lang.get('overlap_percent_label'), minimum=0, maximum=50, value=15, step=1, info=default_lang.get('overlap_percent_info'))
        
        gr.Markdown("**Controle de Influência (Convergência):**")
        with gr.Row():
            handler_strength_slider = gr.Slider(label=default_lang.get('handler_strength_label'), minimum=0.0, maximum=1.0, value=0.5, step=0.05, info=default_lang.get('handler_strength_info'))
            destination_convergence_slider = gr.Slider(label=default_lang.get('destination_convergence_label'), minimum=0.0, maximum=1.0, value=0.75, step=0.05, info=default_lang.get('destination_convergence_info'))
        
        with gr.Accordion(default_lang.get('advanced_accordion_label'), open=False) as advanced_accordion:
             with gr.Row():
                guidance_scale_slider = gr.Slider(label=default_lang.get('guidance_label'), minimum=1.0, maximum=15.0, value=1.0, step=0.5)
                stg_scale_slider = gr.Slider(label=default_lang.get('stg_label'), minimum=0.0, maximum=10.0, value=0.0, step=0.5)
                rescaling_scale_slider = gr.Slider(label=default_lang.get('rescaling_label'), minimum=0.0, maximum=1.0, value=0.15, step=0.05)
             with gr.Row():
                num_inference_steps_slider = gr.Slider(label=default_lang.get('steps_label'), minimum=4, maximum=50, value=7, step=1, info=default_lang.get('steps_info'))
        produce_button = gr.Button(default_lang.get('produce_button'), variant="primary")
    
    video_fragments_gallery = gr.Gallery(label=default_lang.get('video_fragments_gallery_label'), visible=False, object_fit="contain", height="auto", type="filepath")
    final_video_output = gr.Video(label=default_lang.get('final_movie_with_audio_label'), visible=False)

    with gr.Accordion("📝 Log de Geração (Detalhado)", open=False) as log_accordion:
        log_display = gr.Textbox(label="Log da Sessão", lines=20, interactive=False, autoscroll=True)
        update_log_button = gr.Button("Atualizar Log")

    # --- 5. CONEXÕES DA UI ---
    all_ui_components = list(update_ui_language('pt').keys())
    lang_selector.change(fn=update_ui_language, inputs=lang_selector, outputs=all_ui_components)
    
    ref_image_input.upload(fn=preprocess_base_images_wrapper, inputs=ref_image_input, outputs=ref_image_input)

    storyboard_and_keyframes_button.click(
        fn=run_mode_a_wrapper, 
        inputs=[prompt_input, num_keyframes_slider, ref_image_input, resolution_selector, duration_per_fragment_slider], 
        outputs=[storyboard_output, keyframe_gallery, step3_accordion]
    )
    
    storyboard_from_photos_button.click(
        fn=run_mode_b_wrapper,
        inputs=[prompt_input, num_keyframes_slider, ref_image_input],
        outputs=[storyboard_output, keyframe_gallery, step3_accordion]
    )
    
    produce_button.click(
        fn=run_video_production_wrapper,
        inputs=[
            keyframe_gallery, prompt_input, duration_per_fragment_slider, 
            overlap_percent_slider, 
            echo_frames_selector,
            handler_strength_slider,
            destination_convergence_slider,
            guidance_scale_slider, stg_scale_slider, rescaling_scale_slider,
            num_inference_steps_slider,
            resolution_selector, continuity_director_checkbox, cinematographer_checkbox
        ],
        outputs=[video_fragments_gallery, final_video_output]
    )

    update_log_button.click(
        fn=get_log_content,
        inputs=[],
        outputs=[log_display]
    )

if __name__ == "__main__":
    if os.path.exists(WORKSPACE_DIR): 
        logger.info(f"Limpando o workspace anterior em: {WORKSPACE_DIR}")
        shutil.rmtree(WORKSPACE_DIR)
    os.makedirs(WORKSPACE_DIR)
    logger.info(f"Aplicação iniciada. Lançando interface Gradio...")
    demo.queue().launch()