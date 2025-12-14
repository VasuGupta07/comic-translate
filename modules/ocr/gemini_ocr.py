import json
import re
import time
import numpy as np
import requests

from .base import OCREngine
from ..utils.textblock import TextBlock, adjust_text_line_coordinates
from ..utils.translator_utils import MODEL_MAP
from app.ui.settings.settings_page import SettingsPage


class GeminiOCR(OCREngine):
    """OCR engine using Google Gemini models via REST API with batch processing.
    
    This implementation sends the full image with bounding box coordinates in a single
    API call, extracting text for all blocks at once. This significantly reduces API calls
    and avoids rate limiting issues (Error 429).
    """
    
    def __init__(self):
        self.api_key = None
        self.expansion_percentage = 5
        self.model = ''
        self.api_base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.max_output_tokens = 8000
        # Retry configuration for rate limiting
        self.max_retries = 5
        self.base_delay = 1.0  # Base delay in seconds for exponential backoff
        
    def initialize(self, settings: SettingsPage, model: str = 'Gemini-2.5-Pro', 
                   expansion_percentage: int = 5) -> None:
        """
        Initialize the Gemini OCR with API key and parameters.
        
        Args:
            settings: Settings page containing credentials
            model: Gemini model to use for OCR (defaults to Gemini-2.5-Pro)
            expansion_percentage: Percentage to expand text bounding boxes
        """
        self.expansion_percentage = expansion_percentage
        credentials = settings.get_credentials(settings.ui.tr('Google Gemini'))
        self.api_key = credentials.get('api_key', '')
        self.model = MODEL_MAP.get(model)
        
    def process_image(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        """
        Process an image with Gemini-based OCR using batch processing.
        
        Sends the full image with all bounding box coordinates in a single API call,
        which dramatically reduces API calls (from N calls to 1 call per page).
        
        Args:
            img: Input image as numpy array
            blk_list: List of TextBlock objects to update with OCR text
            
        Returns:
            List of updated TextBlock objects with recognized text
        """
        if not blk_list:
            return blk_list
            
        return self._process_batch(img, blk_list)
    
    def _process_batch(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        """
        Process all text blocks in a single API call using batch OCR.
        
        Args:
            img: Input image as numpy array
            blk_list: List of TextBlock objects to update with OCR text
            
        Returns:
            List of updated TextBlock objects with recognized text
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
        
        # Get batch OCR results
        results = self._get_batch_ocr(encoded_img, block_coords)
        
        # Assign results to blocks
        for block_id, text in results.items():
            try:
                idx = int(block_id.replace("block_", ""))
                if 0 <= idx < len(blk_list):
                    blk_list[idx].text = text
            except (ValueError, IndexError):
                continue
                
        return blk_list
    
    def _get_batch_ocr(self, base64_image: str, block_coords: list) -> dict:
        """
        Get OCR results for all blocks in a single API call.
        
        Args:
            base64_image: Base64 encoded full image
            block_coords: List of dicts with block_id and coords
            
        Returns:
            Dictionary mapping block_id to extracted text
        """
        if not self.api_key:
            raise ValueError("API key not initialized. Call initialize() first.")
            
        # Create API endpoint URL
        url = f"{self.api_base_url}/{self.model}:generateContent?key={self.api_key}"
        
        # Build coordinates description for prompt
        coords_json = json.dumps(block_coords, indent=2)
        
        prompt = f"""You are an expert OCR system specialized in reading text from comics, manga, and graphic novels.

I have identified {len(block_coords)} text regions in this image. Each region has a block_id and coordinates [x1, y1, x2, y2] (top-left to bottom-right).

Text regions:
{coords_json}

For EACH text region, extract ALL text exactly as it appears, including:
- Speech bubble dialogue
- Narrative boxes  
- Sound effects (onomatopoeia)
- Signs, labels, or any visible text

Rules:
- Output ONLY valid JSON with the extracted text for each block
- Preserve the original text exactly as written
- For vertical text (common in manga), read top-to-bottom, right-to-left
- If a region has no readable text, use an empty string ""
- Do NOT translate the text
- Do not add descriptions, explanations, or commentary

Respond with ONLY a JSON object in this exact format:
{{
  "block_0": "text from first region",
  "block_1": "text from second region",
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
    
    def _make_request_with_retry(self, url: str, payload: dict) -> dict:
        """
        Make API request with exponential backoff retry for rate limiting.
        
        Args:
            url: API endpoint URL
            payload: Request payload
            
        Returns:
            Dictionary of OCR results
        """
        headers = {"Content-Type": "application/json"}
        
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
                print(f"API error: {response.status_code} {response.text}")
                return {}
        
        print(f"Max retries ({self.max_retries}) exceeded")
        return {}
    
    def _parse_response(self, response_data: dict) -> dict:
        """
        Parse API response and extract OCR results.
        
        Args:
            response_data: Raw API response
            
        Returns:
            Dictionary mapping block_id to text
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