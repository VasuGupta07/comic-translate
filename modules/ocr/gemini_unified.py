import json
import re
import time
import numpy as np
import requests

from .base import OCREngine
from ..utils.textblock import TextBlock, adjust_text_line_coordinates
from ..utils.translator_utils import MODEL_MAP
from app.ui.settings.settings_page import SettingsPage


class GeminiUnifiedProcessor(OCREngine):
    """
    Unified OCR + Translation processor using Google Gemini models.
    
    This implementation combines OCR and translation into a single API call,
    dramatically reducing the number of API calls needed. Instead of making
    separate OCR and translation calls for each image, this batches up to 5
    images per request and returns both source text and translations.
    
    Benefits:
    - Reduces API calls from 20 (10 images × 2 calls) to 2 (10 images ÷ 5 batch size)
    - Significantly reduces rate limiting issues (Error 429)
    - More cost-effective (~$0.003/page vs ~$0.03/page)
    """
    
    # Class constants for configuration
    DEFAULT_MAX_OUTPUT_TOKENS = 8000
    MAX_BATCH_OUTPUT_TOKENS = 16000  # Cap for batched requests to avoid API rejections
    REQUEST_TIMEOUT = 180  # Timeout for API requests in seconds
    
    def __init__(self):
        self.api_key = None
        self.expansion_percentage = 5
        self.model = ''
        self.api_base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.max_output_tokens = self.DEFAULT_MAX_OUTPUT_TOKENS
        self.source_lang = 'Japanese'
        self.target_lang = 'English'
        # Retry configuration for rate limiting - increased for better reliability
        self.max_retries = 8
        self.base_delay = 15.0  # Base delay in seconds for exponential backoff
        self.max_delay = 120.0  # Maximum delay between retries
        
    def initialize(self, settings: SettingsPage, model: str = 'Gemini-2.5-Pro', 
                   expansion_percentage: int = 5,
                   source_lang: str = 'Japanese',
                   target_lang: str = 'English') -> None:
        """
        Initialize the Gemini Unified Processor with API key and parameters.
        
        Args:
            settings: Settings page containing credentials
            model: Gemini model to use (defaults to Gemini-2.5-Pro)
            expansion_percentage: Percentage to expand text bounding boxes
            source_lang: Source language for OCR/translation
            target_lang: Target language for translation
        """
        self.expansion_percentage = expansion_percentage
        credentials = settings.get_credentials(settings.ui.tr('Google Gemini'))
        self.api_key = credentials.get('api_key', '')
        self.model = MODEL_MAP.get(model, 'gemini-2.5-pro')  # Default to gemini-2.5-pro if not found
        self.source_lang = source_lang
        self.target_lang = target_lang
        
    def process_image(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        """
        Process an image with unified OCR + Translation.
        
        This method extracts text and translates it in a single API call.
        
        Args:
            img: Input image as numpy array
            blk_list: List of TextBlock objects to update with OCR text and translations
            
        Returns:
            List of updated TextBlock objects with recognized text AND translations
        """
        if not blk_list:
            return blk_list
            
        return self._process_unified(img, blk_list)
    
    def process_batch(self, images: list[tuple[np.ndarray, list[TextBlock]]]) -> list[list[TextBlock]]:
        """
        Process multiple images in a single batched API call.
        
        This is the key optimization - instead of making separate calls for each
        image's OCR and translation, we batch up to 5 images into a single request.
        
        Args:
            images: List of (image, blk_list) tuples to process
            
        Returns:
            List of updated TextBlock lists, one per input image
        """
        if not images:
            return []
        
        # Process in batches of 5 images
        batch_size = 5
        all_results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = self._process_batch_unified(batch)
            all_results.extend(batch_results)
        
        return all_results
    
    def _process_unified(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        """
        Process a single image with unified OCR + Translation.
        
        Args:
            img: Input image as numpy array
            blk_list: List of TextBlock objects to update
            
        Returns:
            List of updated TextBlock objects with text and translations
        """
        # Build bounding box info for all blocks
        block_coords = []
        for idx, blk in enumerate(blk_list):
            if blk.bubble_xyxy is not None:
                x1, y1, x2, y2 = blk.bubble_xyxy
            else:
                x1, y1, x2, y2 = adjust_text_line_coordinates(
                    blk.xyxy, 
                    self.expansion_percentage, 
                    self.expansion_percentage, 
                    img
                )
            
            # Only include valid coordinates
            if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= img.shape[1] and y2 <= img.shape[0]:
                block_coords.append({
                    "block_id": idx,
                    "coords": [int(x1), int(y1), int(x2), int(y2)]
                })
        
        if not block_coords:
            return blk_list
        
        # Encode full image
        encoded_img = self.encode_image(img)
        
        # Get unified OCR + Translation results
        results = self._get_unified_results(encoded_img, block_coords)
        
        # Assign results to blocks
        for block_id, data in results.items():
            try:
                idx = int(block_id.replace("block_", ""))
                if 0 <= idx < len(blk_list):
                    if isinstance(data, dict):
                        blk_list[idx].text = data.get("source", "")
                        blk_list[idx].translation = data.get("translation", "")
                    else:
                        # Fallback for simple text response
                        blk_list[idx].text = str(data)
            except (ValueError, IndexError):
                continue
                
        return blk_list
    
    def _process_batch_unified(self, batch: list[tuple[np.ndarray, list[TextBlock]]]) -> list[list[TextBlock]]:
        """
        Process a batch of images in a single API call.
        
        Args:
            batch: List of (image, blk_list) tuples (max 5)
            
        Returns:
            List of updated TextBlock lists
        """
        if not self.api_key:
            raise ValueError("API key not initialized. Call initialize() first.")
        
        # Prepare all images and their coordinates
        image_data = []
        for img_idx, (img, blk_list) in enumerate(batch):
            block_coords = []
            for blk_idx, blk in enumerate(blk_list):
                if blk.bubble_xyxy is not None:
                    x1, y1, x2, y2 = blk.bubble_xyxy
                else:
                    x1, y1, x2, y2 = adjust_text_line_coordinates(
                        blk.xyxy, 
                        self.expansion_percentage, 
                        self.expansion_percentage, 
                        img
                    )
                
                if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= img.shape[1] and y2 <= img.shape[0]:
                    block_coords.append({
                        "block_id": blk_idx,
                        "coords": [int(x1), int(y1), int(x2), int(y2)]
                    })
            
            if block_coords:
                image_data.append({
                    "image_idx": img_idx,
                    "encoded_img": self.encode_image(img),
                    "block_coords": block_coords,
                    "blk_list": blk_list
                })
        
        if not image_data:
            return [item[1] for item in batch]  # Return original blk_lists
        
        # Make batched API call
        results = self._make_batch_request(image_data)
        
        # Apply results to blk_lists
        result_blk_lists = []
        for item in batch:
            result_blk_lists.append(item[1])  # Start with original
            
        for img_key, img_results in results.items():
            try:
                img_idx = int(img_key.replace("image_", ""))
                if img_idx < len(batch):
                    blk_list = batch[img_idx][1]
                    for block_key, data in img_results.items():
                        try:
                            blk_idx = int(block_key.replace("block_", ""))
                            if 0 <= blk_idx < len(blk_list):
                                if isinstance(data, dict):
                                    blk_list[blk_idx].text = data.get("source", "")
                                    blk_list[blk_idx].translation = data.get("translation", "")
                                else:
                                    blk_list[blk_idx].text = str(data)
                        except (ValueError, IndexError):
                            continue
            except (ValueError, IndexError):
                continue
        
        return result_blk_lists
    
    def _get_unified_results(self, base64_image: str, block_coords: list) -> dict:
        """
        Get unified OCR + Translation results for a single image.
        
        Args:
            base64_image: Base64 encoded image
            block_coords: List of dicts with block_id and coords
            
        Returns:
            Dictionary mapping block_id to {source, translation}
        """
        if not self.api_key:
            raise ValueError("API key not initialized. Call initialize() first.")
            
        # Create API endpoint URL
        url = f"{self.api_base_url}/{self.model}:generateContent?key={self.api_key}"
        
        # Build coordinates description for prompt
        coords_json = json.dumps(block_coords, indent=2)
        
        prompt = f"""You are an expert OCR and translation system specialized in comics, manga, and graphic novels.

I have identified {len(block_coords)} text regions in this image. Each region has a block_id and coordinates [x1, y1, x2, y2] (top-left to bottom-right).

Text regions:
{coords_json}

For EACH text region:
1. EXTRACT the original text exactly as it appears (OCR)
2. TRANSLATE the text from {self.source_lang} to {self.target_lang}

Important rules:
- Output ONLY valid JSON with source and translation for each block
- For vertical text (common in manga), read top-to-bottom, right-to-left
- If a region has no readable text, use empty strings
- Make translations natural and conversational
- Preserve the tone, emotion, and personality
- Keep translations concise to fit in speech bubbles
- Do NOT add descriptions or commentary

Respond with ONLY a JSON object in this exact format:
{{
  "block_0": {{"source": "original text", "translation": "translated text"}},
  "block_1": {{"source": "original text", "translation": "translated text"}},
  ...
}}"""

        # Setup generation config
        generation_config = {
            "maxOutputTokens": self.max_output_tokens,
        }
        
        # Setup safety settings to disable all content filtering
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        payload = {
            "contents": [{
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image
                        }
                    },
                    {
                        "text": prompt
                    }
                ]
            }],
            "generationConfig": generation_config,
            "safetySettings": safety_settings,
        }
        
        # Make request with retry logic for rate limiting
        return self._make_request_with_retry(url, payload)
    
    def _make_batch_request(self, image_data: list) -> dict:
        """
        Make a batched API request for multiple images.
        
        Args:
            image_data: List of image data dicts with encoded_img and block_coords
            
        Returns:
            Dictionary mapping image_idx to block results
        """
        url = f"{self.api_base_url}/{self.model}:generateContent?key={self.api_key}"
        
        # Build multi-image prompt
        parts = []
        image_descriptions = []
        
        for data in image_data:
            img_idx = data["image_idx"]
            coords_json = json.dumps(data["block_coords"], indent=2)
            
            # Add image
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": data["encoded_img"]
                }
            })
            
            image_descriptions.append(f"""
Image {img_idx}:
Text regions: {coords_json}""")
        
        # Add the unified prompt
        prompt = f"""You are an expert OCR and translation system specialized in comics, manga, and graphic novels.

I have {len(image_data)} images. For each image, I've identified text regions with coordinates [x1, y1, x2, y2] (top-left to bottom-right).

{''.join(image_descriptions)}

For EACH text region in EACH image:
1. EXTRACT the original text exactly as it appears (OCR)
2. TRANSLATE the text from {self.source_lang} to {self.target_lang}

Important rules:
- Output ONLY valid JSON with source and translation for each block in each image
- For vertical text (common in manga), read top-to-bottom, right-to-left
- If a region has no readable text, use empty strings
- Make translations natural and conversational
- Preserve the tone, emotion, and personality
- Keep translations concise to fit in speech bubbles
- Do NOT add descriptions or commentary

Respond with ONLY a JSON object in this exact format:
{{
  "image_0": {{
    "block_0": {{"source": "original text", "translation": "translated text"}},
    "block_1": {{"source": "original text", "translation": "translated text"}}
  }},
  "image_1": {{
    "block_0": {{"source": "original text", "translation": "translated text"}}
  }}
}}"""
        
        parts.append({"text": prompt})
        
        # Calculate output tokens with a cap to avoid API rejections
        batch_tokens = min(self.max_output_tokens * len(image_data), self.MAX_BATCH_OUTPUT_TOKENS)
        generation_config = {
            "maxOutputTokens": batch_tokens,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": generation_config,
            "safetySettings": safety_settings,
        }
        
        return self._make_request_with_retry(url, payload)
    
    def _make_request_with_retry(self, url: str, payload: dict) -> dict:
        """
        Make API request with exponential backoff retry for rate limiting.
        
        Args:
            url: API endpoint URL
            payload: Request payload
            
        Returns:
            Dictionary of results
        """
        headers = {"Content-Type": "application/json"}
        
        for attempt in range(self.max_retries):
            response = requests.post(
                url,
                headers=headers, 
                json=payload,
                timeout=self.REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                return self._parse_response(response.json())
            elif response.status_code == 429:
                # Rate limited - apply exponential backoff
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                print(f"Rate limited (429). Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(delay)
            elif response.status_code >= 500:
                # Server error - retry with backoff
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                print(f"Server error ({response.status_code}). Retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(delay)
            else:
                # Other errors - don't retry
                print(f"API error: {response.status_code} {response.text}")
                return {}
        
        print(f"Max retries ({self.max_retries}) exceeded")
        return {}
    
    def _parse_response(self, response_data: dict) -> dict:
        """
        Parse API response and extract results.
        
        Args:
            response_data: Raw API response
            
        Returns:
            Dictionary mapping block_id to {source, translation} or image_id to block results
        """
        candidates = response_data.get("candidates", [])
        if not candidates:
            return {}
            
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        
        # Concatenate all text parts
        result_text = ""
        for part in parts:
            if "text" in part:
                result_text += part["text"]
        
        # Extract JSON from response
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from response: {result_text[:200]}")
        
        return {}
