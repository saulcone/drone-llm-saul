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
        self.center_threshold = 50  # Píxeles de desviación permitidos
        self.width_threshold_enter = 0.35 # Si ocupa el 35% del ancho, está cerca

    def get_correction(self, image_cv2):
        """
        Pide a Gemini las coordenadas de la puerta abierta y calcula la corrección.
        """
        height, width, _ = image_cv2.shape
        img_center_x = width // 2

        # Convertir imagen para Gemini
        im_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
        
        # PROMPT MEJORADO: Especificamos explícitamente qué IGNORAR.
        prompt = (
            "Detect the OPEN door used for passage. "
            "Ignore closed doors, cabinets, closets, wardrobes, and windows. "
            "Only detect a door if it is clearly OPEN and leads to another room. "
            "Return the bounding box in [ymin, xmin, ymax, xmax] format. "
            "If no open door is clearly visible, return empty JSON."
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
                            "box_2d": {
                                "type": "array",
                                "items": {"type": "integer"} # [ymin, xmin, ymax, xmax]
                            },
                            "label": {"type": "string"}
                        }
                    },
                    temperature=0.0
                )
            )
            
            res = json.loads(response.text)
            
            # Verificar si detectó algo
            if not res or "box_2d" not in res or not res["box_2d"]:
                return None, "No door detected by Gemini"

            # Extraer coordenadas y escalar si es necesario
            ymin, xmin, ymax, xmax = res["box_2d"]
            
            if xmax <= 1000 and ymax <= 1000:
                scale_x = width / 1000
                xmin = int(xmin * scale_x)
                xmax = int(xmax * scale_x)

            door_center_x = (xmin + xmax) // 2
            door_width = xmax - xmin
            offset = door_center_x - img_center_x

            # --- Lógica de Seguridad ---
            
            # 1. Alineación Lateral
            if abs(offset) > self.center_threshold:
                if offset < 0:
                    return "D1", f"Gemini Safety: Door offset {offset}px LEFT"
                else:
                    return "D2", f"Gemini Safety: Door offset {offset}px RIGHT"

            # 2. Entrada Franca
            if door_width > (width * self.width_threshold_enter):
                return "A1", "Gemini Safety: Door centered and close. Force Entry."

            return None, "Door aligned but far. Gemini VLM decides."

        except Exception as e:
            print(f"Error in DoorDetector: {e}")
            return None, "Error in detection"