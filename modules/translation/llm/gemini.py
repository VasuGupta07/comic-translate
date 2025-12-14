from typing import Any
import time
import numpy as np
import requests

from .base import BaseLLMTranslation
from ...utils.translator_utils import MODEL_MAP


class GeminiTranslation(BaseLLMTranslation):
    """Translation engine using Google Gemini models via REST API with retry logic."""
    
    def __init__(self):
        super().__init__()
        self.model_name = None
        self.api_key = None
        self.api_base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        # Retry configuration for rate limiting
        self.max_retries = 5
        self.base_delay = 1.0  # Base delay in seconds for exponential backoff
    
    def initialize(self, settings: Any, source_lang: str, target_lang: str, model_name: str, **kwargs) -> None:
        """
        Initialize Gemini translation engine.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            model_name: Gemini model name
        """
        super().initialize(settings, source_lang, target_lang, **kwargs)
        
        self.model_name = model_name
        credentials = settings.get_credentials(settings.ui.tr('Google Gemini'))
        self.api_key = credentials.get('api_key', '')
        
        # Map friendly model name to API model name
        self.model = MODEL_MAP.get(self.model_name)
    
    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        Perform translation using Gemini REST API with retry logic for rate limiting.
        
        Args:
            user_prompt: The prompt to send to the model
            system_prompt: System instructions for the model
            image: Image data as numpy array
            
        Returns:
            Translated text from the model
        """
        # Create API endpoint URL
        url = f"{self.api_base_url}/{self.model}:generateContent?key={self.api_key}"
        
        # Setup generation config
        generation_config = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_tokens,
            "topP": self.top_p,
        }
        
        # Setup safety settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # Prepare parts for the request
        parts = []
        
        # Add image if needed
        if self.img_as_llm_input:
            # Base64 encode the image

            img_b64, mime_type = self.encode_image(image)
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": img_b64
                }
            })
        
        # Add text prompt
        parts.append({"text": user_prompt})
        
        # Create the request payload
        payload = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": generation_config,
            "safetySettings": safety_settings
        }
        
        # Add system instructions if provided
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        
        # Send request with retry logic
        headers = {
            "Content-Type": "application/json"
        }
        
        return self._make_request_with_retry(url, headers, payload)
    
    def _make_request_with_retry(self, url: str, headers: dict, payload: dict) -> str:
        """
        Make API request with exponential backoff retry for rate limiting.
        
        Args:
            url: API endpoint URL
            headers: Request headers
            payload: Request payload
            
        Returns:
            Translated text from the model
        """
        for attempt in range(self.max_retries):
            response = requests.post(
                url, 
                headers=headers, 
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                return self._parse_response(response.json())
            elif response.status_code == 429:
                # Rate limited - apply exponential backoff
                delay = self.base_delay * (2 ** attempt)
                print(f"Rate limited (429). Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(delay)
            elif response.status_code >= 500:
                # Server error - retry with backoff
                delay = self.base_delay * (2 ** attempt)
                print(f"Server error ({response.status_code}). Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(delay)
            else:
                # Other errors - don't retry
                error_msg = f"API request failed with status code {response.status_code}: {response.text}"
                raise Exception(error_msg)
        
        raise Exception(f"Max retries ({self.max_retries}) exceeded for Gemini API")
    
    def _parse_response(self, response_data: dict) -> str:
        """
        Parse API response and extract translated text.
        
        Args:
            response_data: Raw API response
            
        Returns:
            Translated text
        """
        candidates = response_data.get("candidates", [])
        if not candidates:
            return "No response generated"
        
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        
        # Concatenate all text parts
        result = ""
        for part in parts:
            if "text" in part:
                result += part["text"]
        
        return result