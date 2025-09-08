# managers/gemini_manager.py
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
# PENDING PATENT NOTICE: Please see NOTICE.md.
#
# Version: 1.1.1
#
# This file defines the GeminiManager, a specialist responsible for raw communication
# with the Google Gemini API. It acts as a lean API client, handling requests,
# parsing responses, and managing API-level errors. It does not contain any
# high-level prompt engineering or creative logic.

import os
import logging
import json
from pathlib import Path
import gradio as gr
from PIL import Image
import google.generativeai as genai
import re
from typing import List, Union, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def robust_json_parser(raw_text: str) -> dict:
    """
    Parses a JSON object from a string that might contain extra text,
    such as Markdown code blocks from an LLM's response.
    """
    clean_text = raw_text.strip()
    try:
        match = re.search(r'```json\s*(\{.*?\})\s*```', clean_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        
        start_index = clean_text.find('{')
        end_index = clean_text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = clean_text[start_index : end_index + 1]
            return json.loads(json_str)
        else:
            raise ValueError("No valid JSON object could be found in the AI's response.")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON. The AI returned the following text:\n---\n{raw_text}\n---")
        raise ValueError(f"The AI returned an invalid JSON format: {e}")

class GeminiManager:
    """
    Manages raw interactions with the Google Gemini API.
    """
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-pro') 
            logger.info("GeminiManager (Communication Layer) initialized successfully.")
        else:
            self.model = None
            logger.warning("Gemini API key not found. GeminiManager disabled.")
            
    def _check_model(self):
        """Raises an error if the Gemini API is not configured."""
        if not self.model:
            raise gr.Error("The Google Gemini API key is not configured (GEMINI_API_KEY).")

    def _generate_content(self, prompt_parts: List[Any]) -> str:
        """Internal method to make the API call."""
        self._check_model()
        logger.info("Calling Gemini API...")
        response = self.model.generate_content(prompt_parts)
        logger.info(f"Gemini responded with raw text: {response.text}")
        return response.text

    def get_raw_text(self, prompt_parts: List[Any]) -> str:
        """
        Sends a prompt to the Gemini API and returns the raw text response.

        Args:
            prompt_parts (List[Any]): A list containing strings and/or PIL.Image objects.

        Returns:
            str: The raw string response from the API.
        """
        try:
            return self._generate_content(prompt_parts)
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}", exc_info=True)
            raise gr.Error(f"Gemini API communication failed: {e}")

    def get_json_object(self, prompt_parts: List[Any]) -> dict:
        """
        Sends a prompt to the Gemini API, expects a JSON response, parses it, and returns a dictionary.

        Args:
            prompt_parts (List[Any]): A list containing strings and/or PIL.Image objects.

        Returns:
            dict: The parsed JSON object from the API response.
        """
        try:
            raw_response = self._generate_content(prompt_parts)
            return robust_json_parser(raw_response)
        except Exception as e:
            logger.error(f"Gemini API call or JSON parsing failed: {e}", exc_info=True)
            raise gr.Error(f"Gemini API communication or response parsing failed: {e}")

# --- Singleton Instance ---
gemini_manager_singleton = GeminiManager()