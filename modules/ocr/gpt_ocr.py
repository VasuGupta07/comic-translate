import re
import time
import numpy as np
import requests
import json

from .base import OCREngine
from ..utils.textblock import TextBlock, adjust_text_line_coordinates
from ..utils.translator_utils import MODEL_MAP


class GPTOCR(OCREngine):
    """OCR engine using GPT vision capabilities via direct REST API calls with batch processing.
    
    This implementation sends the full image with bounding box coordinates in a single
    API call, extracting text for all blocks at once. This significantly reduces API calls
    and avoids rate limiting issues.
    """
    
    def __init__(self):
        self.api_key = None
        self.expansion_percentage = 0
        self.model = None
        self.api_base_url = 'https://api.openai.com/v1/chat/completions'
        self.max_tokens = 8000
        # Retry configuration for rate limiting
        self.max_retries = 5
        self.base_delay = 1.0  # Base delay in seconds for exponential backoff
        
    def initialize(self, api_key: str, model: str = 'GPT-4.1-mini', 
                  expansion_percentage: int = 0) -> None:
        """
        Initialize the GPT OCR with API key and parameters.
        
        Args:
            api_key: OpenAI API key for authentication
            model: GPT model to use for OCR (defaults to gpt-4o)
            expansion_percentage: Percentage to expand text bounding boxes
        """
        self.api_key = api_key
        self.model = MODEL_MAP.get(model)
        self.expansion_percentage = expansion_percentage
        
    def process_image(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        """
        Process an image with GPT-based OCR using batch processing.
        
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

        # Prepare request headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_completion_tokens": self.max_tokens
        }
        
        # Make request with retry logic
        return self._make_request_with_retry(payload, headers)
    
    def _make_request_with_retry(self, payload: dict, headers: dict) -> dict:
        """
        Make API request with exponential backoff retry for rate limiting.
        
        Args:
            payload: Request payload
            headers: Request headers
            
        Returns:
            Dictionary of OCR results
        """
        for attempt in range(self.max_retries):
            response = requests.post(
                self.api_base_url,
                headers=headers,
                data=json.dumps(payload),
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
        try:
            result_text = response_data['choices'][0]['message']['content']
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                return json.loads(json_match.group(0))
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Failed to parse response: {e}")
        
        return {}