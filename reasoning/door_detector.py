import os
import cv2
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image
import numpy as np

load_dotenv()

class DoorDetector:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name
        
        # Umbrales
        self.center_threshold = 50  
        self.width_threshold_enter = 0.35 

    def get_correction(self, image_cv2):
        height, width, _ = image_cv2.shape
        img_center_x = width // 2

        im_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
        
        # PROMPT BLINDADO: Obligamos a pensar primero si es transitable.
        prompt = (
            "Analyze the image. Is there an OPEN, passable doorway leading to another room? "
            "If it is a closed door, a cabinet, a closet, or just a wall, set 'is_open_door_present' to false. "
            "Only if a clearly open gap/void is visible, set 'is_open_door_present' to true and provide its 'box_2d' [ymin, xmin, ymax, xmax]."
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, im_pil],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "is_open_door_present": {"type": "boolean"},
                            "box_2d": {
                                "type": "array",
                                "items": {"type": "integer"}
                            },
                            "label": {"type": "string"}
                        }
                    },
                    temperature=0.0
                )
            )
            
            res = json.loads(response.text)
            
            # FILTRO DE SEGURIDAD: Si dice que no hay puerta, abortamos la detección
            if not res.get("is_open_door_present", False) or not res.get("box_2d"):
                return None, "No open door detected by Gemini (Ignored)"

            ymin, xmin, ymax, xmax = res["box_2d"]
            
            if xmax <= 1000 and ymax <= 1000:
                scale_x = width / 1000
                xmin = int(xmin * scale_x)
                xmax = int(xmax * scale_x)

            door_center_x = (xmin + xmax) // 2
            door_width = xmax - xmin
            offset = door_center_x - img_center_x
            
            # --- Lógica de Seguridad Mejorada ---

            # Si el offset es masivo (ej. > 150px), NO usar D1/D2.
            # Es más seguro que el VLM rote (B1/C1).
            if abs(offset) > 150:
                return None, f"Offset too large ({offset}px). Let VLM rotate instead of sliding."

            # 1. Alineación Lateral (Solo para ajustes finos)
            if abs(offset) > self.center_threshold:
                if offset < 0:
                    return "D1", f"Gemini Safety: Small offset {offset}px LEFT"
                else:
                    return "D2", f"Gemini Safety: Small offset {offset}px RIGHT"

            # 2. Entrada Franca
            if door_width > (width * self.width_threshold_enter):
                return "A1", "Gemini Safety: Door centered and close. Force Entry."

            return None, "Door aligned but far. Gemini VLM decides."

        except Exception as e:
            print(f"Error in DoorDetector: {e}")
            return None, "Error in detection"