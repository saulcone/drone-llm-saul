import os
import json
import logging
import cv2
import ollama

from reasoning.load_prompts import load_system_prompt
from reasoning.VLMModel import VLMBaseModel

logger = logging.getLogger(__name__)

class OllamaDescriptor(VLMBaseModel):
    def __init__(self, model="qwen3-vl:235b-cloud", temperature=0.0):
        # Asegúrate de haber hecho: pip install ollama
        self.model = model
        self.temperature = temperature

    def generate_trajectory(self, image, user_query, topology=None, state=None,
                            previous_movement=None, mov_history=None, test=False):
        
        # Modo de prueba rápido para depuración
        if test:
            return {
                "room": "hallway",
                "movement": "B3",
                "state": "Oriented Towards Door",
                "description": "Visible hallway with a door, move robot to face the door"
            }

        # 1. Codificar la imagen a Base64 (usando la función de tu clase padre VLMBaseModel)
        # Opcional: Si el modelo 235B va exageradamente lento, puedes descomentar la siguiente línea
        # para reducir la resolución de la imagen antes de enviarla, aligerando el cálculo visual:
        # image = cv2.resize(image, (240, 180))
        img_b64_str = self.encode_image_base64(image)
        
        # 2. Construir el contexto del usuario (Inyectando la Memoria)
        user_text_parts = [user_query]
        
        if state:
            user_text_parts.append("Current FSM state: " + state)
        if topology:
            user_text_parts.append("Topology: " + json.dumps(topology))
        if previous_movement:
            user_text_parts.append("Previous movement command: " + previous_movement)
        if mov_history:
            user_text_parts.append("Movement history (last commands): " + str(mov_history))
            
        user_text = "\n\n".join(user_text_parts)

        # 3. Cargar los prompts de sistema y de estado
        system_prompt = load_system_prompt(
            path_dir="prompts",
            system_prompt_path="drone_system_prompt.txt",
            output_prompt_path="output_prompt.txt",
            curr_state=state
        )

        print("Current FSM state: " + str(state))

        # 4. Formatear los mensajes para la librería de Ollama
        # Ollama acepta directamente la imagen en base64 dentro de la lista 'images'
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user", 
                "content": user_text,
                "images": [img_b64_str]
            }
        ]        

        try:
            # 5. Llamada a la API local de Ollama (Servicio en segundo plano)
            response = ollama.chat(
                model=self.model,
                messages=messages,
                format='json', # Forzamos la salida en JSON
                options={
                    "temperature": self.temperature
                }
            )
            
            # 6. Parseo seguro de la respuesta JSON
            response_text = response['message']['content'].strip()
            
            # Limpieza de bloques markdown (muy común en modelos Qwen locales)
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()

            return json.loads(response_text)

        except Exception as e:
            logger.exception(f"Error calling Ollama model {self.model}: {str(e)}")
            return None