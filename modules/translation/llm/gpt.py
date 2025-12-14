from typing import Any
import time
import numpy as np
import requests
import json

from .base import BaseLLMTranslation
from ...utils.translator_utils import MODEL_MAP


class GPTTranslation(BaseLLMTranslation):
    """Translation engine using OpenAI GPT models through direct REST API calls with retry logic."""
    
    def __init__(self):
        super().__init__()
        self.model_name = None
        self.api_key = None
        self.api_base_url = "https://api.openai.com/v1"
        self.supports_images = True
        # Retry configuration for rate limiting
        self.max_retries = 5
        self.base_delay = 1.0  # Base delay in seconds for exponential backoff
    
    def initialize(self, settings: Any, source_lang: str, target_lang: str, model_name: str, **kwargs) -> None:
        """
        Initialize GPT translation engine.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            model_name: GPT model name
        """
        super().initialize(settings, source_lang, target_lang, **kwargs)
        
        self.model_name = model_name
        credentials = settings.get_credentials(settings.ui.tr('Open AI GPT'))
        self.api_key = credentials.get('api_key', '')
        self.model = MODEL_MAP.get(self.model_name)
    
    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        Perform translation using direct REST API calls to OpenAI with retry logic.
        
        Args:
            user_prompt: Text prompt from user
            system_prompt: System instructions
            image: Image as numpy array
            
        Returns:
            Translated text
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        if self.supports_images and self.img_as_llm_input:
            # Use the base class method to encode the image
            encoded_image, mime_type = self.encode_image(image)
            
            messages = [
                {
                    "role": "system", 
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}}
                    ]
                }
            ]
        else:
            messages = [
                {
                    "role": "system", 
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user", 
                    "content": [{"type": "text", "text": user_prompt}]
                }
            ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

        return self._make_api_request(payload, headers)
    
    def _make_api_request(self, payload: dict, headers: dict) -> str:
        """
        Make API request with exponential backoff retry for rate limiting.
        
        Args:
            payload: Request payload
            headers: Request headers
            
        Returns:
            Translated text from the model
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_base_url}/chat/completions",
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=120
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    return response_data["choices"][0]["message"]["content"]
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
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    error_msg = f"API request failed: {str(e)}"
                    if hasattr(e, 'response') and e.response is not None:
                        try:
                            error_details = e.response.json()
                            error_msg += f" - {json.dumps(error_details)}"
                        except Exception:
                            error_msg += f" - Status code: {e.response.status_code}"
                    raise RuntimeError(error_msg)
                # Retry on connection errors
                delay = self.base_delay * (2 ** attempt)
                print(f"Connection error. Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(delay)
        
        raise RuntimeError(f"Max retries ({self.max_retries}) exceeded for OpenAI API")