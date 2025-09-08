# engineers/deformes3D_thinker.py
#
# Copyright (C) 2025 Carlos Rodrigues dos Santos
#
# Version: 4.0.0 (Definitive)
#
# This is the definitive, robust implementation. It directly contains the prompt
# enhancement logic copied from the LTX pipeline's utils. It accesses the
# enhancement models loaded by the LTX Manager and performs the captioning
# and LLM generation steps locally, ensuring full control and compatibility.

import logging
from PIL import Image
import torch

# Importa o singleton do LTX para ter acesso à sua pipeline e aos modelos nela
from managers.ltx_manager import ltx_manager_singleton

# Importa o prompt de sistema do LTX para garantir consistência
from ltx_video.utils.prompt_enhance_utils import I2V_CINEMATIC_PROMPT

logger = logging.getLogger(__name__)

class Deformes3DThinker:
    """
    The tactical specialist that now directly implements the prompt enhancement
    logic, using the models provided by the LTX pipeline.
    """
    
    def __init__(self):
        # Acessa a pipeline exposta para obter os modelos necessários
        pipeline = ltx_manager_singleton.prompt_enhancement_pipeline
        if not pipeline:
            raise RuntimeError("Deformes3DThinker could not access the LTX pipeline.")
        
        # Armazena os modelos e processadores como atributos diretos
        self.caption_model = pipeline.prompt_enhancer_image_caption_model
        self.caption_processor = pipeline.prompt_enhancer_image_caption_processor
        self.llm_model = pipeline.prompt_enhancer_llm_model
        self.llm_tokenizer = pipeline.prompt_enhancer_llm_tokenizer
        
        # Verifica se os modelos foram realmente carregados
        if not all([self.caption_model, self.caption_processor, self.llm_model, self.llm_tokenizer]):
            logger.warning("Deformes3DThinker initialized, but one or more enhancement models were not loaded by the LTX pipeline. Fallback will be used.")
        else:
            logger.info("Deformes3DThinker initialized and successfully linked to LTX enhancement models.")

    @torch.no_grad()
    def get_enhanced_motion_prompt(self, global_prompt: str, story_history: str, 
                                   past_keyframe_path: str, present_keyframe_path: str, future_keyframe_path: str,
                                   past_scene_desc: str, present_scene_desc: str, future_scene_desc: str) -> str:
        """
        Generates a refined motion prompt by directly executing the enhancement pipeline logic.
        """
        # Verifica se os modelos estão disponíveis antes de tentar usá-los
        if not all([self.caption_model, self.caption_processor, self.llm_model, self.llm_tokenizer]):
            logger.warning("Enhancement models not available. Using fallback prompt.")
            return f"A cinematic transition from '{present_scene_desc}' to '{future_scene_desc}'."

        try:
            present_image = Image.open(present_keyframe_path).convert("RGB")
            
            # --- INÍCIO DA LÓGICA COPIADA E ADAPTADA DO LTX ---
            
            # 1. Gerar a caption da imagem de referência (presente)
            image_captions = self._generate_image_captions([present_image])
            
            # 2. Construir o prompt para o LLM
            # Usamos a cena futura como o "prompt do usuário"
            messages = [
                {"role": "system", "content": I2V_CINEMATIC_PROMPT},
                {"role": "user", "content": f"user_prompt: {future_scene_desc}\nimage_caption: {image_captions[0]}"},
            ]

            # 3. Gerar e decodificar o prompt final com o LLM
            enhanced_prompt = self._generate_and_decode_prompts(messages)
            
            # --- FIM DA LÓGICA COPIADA E ADAPTADA ---

            logger.info(f"Deformes3DThinker received enhanced prompt: '{enhanced_prompt}'")
            return enhanced_prompt

        except Exception as e:
            logger.error(f"The Film Director (Deformes3D Thinker) failed during enhancement: {e}. Using fallback.", exc_info=True)
            return f"A smooth, continuous cinematic transition from '{present_scene_desc}' to '{future_scene_desc}'."

    def _generate_image_captions(self, images: list[Image.Image]) -> list[str]:
        """
        Lógica interna para gerar captions, copiada do LTX utils.
        """
        # O modelo Florence-2 do LTX não usa um system_prompt aqui, mas um task_prompt
        task_prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.caption_processor(
            text=[task_prompt] * len(images), images=images, return_tensors="pt"
        ).to(self.caption_model.device)

        generated_ids = self.caption_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        
        # Usa o post_process_generation para extrair a resposta limpa
        generated_text = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        processed_result = self.caption_processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(images[0].width, images[0].height)
        )
        return [processed_result[task_prompt]]

    def _generate_and_decode_prompts(self, messages: list[dict]) -> str:
        """
        Lógica interna para gerar prompt com o LLM, copiada do LTX utils.
        """
        text = self.llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.llm_model.device)

        output_ids = self.llm_model.generate(**model_inputs, max_new_tokens=256)
        
        input_ids_len = model_inputs.input_ids.shape[1]
        decoded_prompts = self.llm_tokenizer.batch_decode(
            output_ids[:, input_ids_len:], skip_special_tokens=True
        )
        return decoded_prompts[0].strip()

# --- Singleton Instantiation ---
try:
    deformes3d_thinker_singleton = Deformes3DThinker()
except Exception as e:
    # A falha já terá sido logada dentro do __init__
    deformes3d_thinker_singleton = None
    raise e