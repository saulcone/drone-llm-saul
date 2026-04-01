import os
import cv2
import json
import base64
import requests
from PIL import Image
import numpy as np
from io import BytesIO

class DoorDetectorOllama:
    def __init__(self, model_name="llava:7b", ollama_url="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        
        # Umbrales
        self.center_threshold = 20  
        self.width_threshold_enter = 0.35 

    def _image_to_base64(self, image_cv2):
        """Convierte imagen OpenCV a base64 para Ollama"""
        im_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        im_pil.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _call_ollama(self, prompt, image_b64):
        """Llama a la API de Ollama con imagen"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def _parse_response(self, response_text):
        """Parsea la respuesta de Ollama extrayendo JSON"""
        # Intentar extraer JSON de la respuesta
        try:
            # Buscar JSON en la respuesta
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Parseo manual si no hay JSON válido
        response_lower = response_text.lower()
        result = {
            "is_open_door_present": False,
            "box_2d": None,
            "is_perpendicular": False
        }
        
        # Detectar si hay puerta abierta
        if any(phrase in response_lower for phrase in ["open door", "doorway", "puerta abierta", "yes"]):
            if "no " not in response_lower[:50] and "closed" not in response_lower:
                result["is_open_door_present"] = True
        
        # Detectar perpendicularidad
        if any(phrase in response_lower for phrase in ["perpendicular", "straight", "square", "facing directly"]):
            result["is_perpendicular"] = True
            
        return result

    def get_correction(self, image_cv2, current_state=""):
        # Solo actuar en estados de alineación/cruce
        if current_state not in ["Oriented Towards Door", "Go Through Door"]:
            return None, "Safety Guard restricted to alignment states."

        height, width, _ = image_cv2.shape
        img_center_x = width // 2

        image_b64 = self._image_to_base64(image_cv2)
        
        # Prompt optimizado para LLaVA
        prompt = """Analyze this image for door detection. Answer in JSON format ONLY.

Look for an OPEN doorway (not closed door, not cabinet, not closet).
If multiple doors exist, pick the one closest to image center.

Respond with ONLY this JSON (no other text):
{
    "is_open_door_present": true/false,
    "box_2d": [ymin, xmin, ymax, xmax] or null,
    "is_perpendicular": true/false,
    "label": "description"
}

For box_2d, estimate pixel coordinates where: top-left is (0,0), image is approximately 640x480.
is_perpendicular: true only if camera faces doorway straight-on (not angled)."""

        try:
            response_text = self._call_ollama(prompt, image_b64)
            res = self._parse_response(response_text)
            
            # Si no hay puerta detectada
            if not res.get("is_open_door_present", False):
                if current_state == "Go Through Door":
                    return "B2", "Ollama Safety: BLIND SPOT/WALL IMMINENT. Emergency abort."
                return None, "No open door detected by Ollama (Ignored)"

            box = res.get("box_2d")
            if box and len(box) == 4:
                ymin, xmin, ymax, xmax = box
                
                # Escalar si las coordenadas están normalizadas (0-1000)
                if xmax <= 1000 and ymax <= 1000 and xmax > 1:
                    scale_x = width / 1000
                    xmin = int(xmin * scale_x)
                    xmax = int(xmax * scale_x)

                door_center_x = (xmin + xmax) // 2
                door_width = xmax - xmin
                offset = door_center_x - img_center_x
            else:
                # Sin bounding box, usar heurística de centro
                offset = 0
                door_width = width * 0.5

            if current_state == "Oriented Towards Door":
                perp = res.get("is_perpendicular", False)
                centered = abs(offset) <= self.center_threshold
                return None, f"Ollama Safety (info only): perpendicular={perp}, centered={centered}, offset={offset}px. VLM decides."

            if current_state == "Go Through Door":
                if abs(offset) <= self.center_threshold and door_width > (width * self.width_threshold_enter):
                    return "A1", "Ollama Safety: Door centered and wide. Continue crossing."
                return None, "Go Through Door: VLM decides alignment."

        except requests.exceptions.ConnectionError:
            print("Error: Ollama no está corriendo. Ejecuta 'ollama serve' primero.")
            return None, "Ollama connection error"
        except Exception as e:
            print(f"Error in DoorDetectorOllama: {e}")
            return None, "Error in detection"
