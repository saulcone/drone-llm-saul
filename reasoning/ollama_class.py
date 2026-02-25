import os
import json
import logging
import base64
from openai import OpenAI
from reasoning.load_prompts import load_system_prompt
from reasoning.VLMModel import VLMBaseModel

logger = logging.getLogger(__name__)

class OllamaDescriptor(VLMBaseModel):
    def __init__(self, model="qwen3-vl:8b", temperature=0.0):
        # Configuramos el cliente para apuntar al túnel SSH en localhost
        self.client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama' # Requerido por la librería de OpenAI, aunque Ollama no lo use
        )
        self.model = model
        self.temperature = temperature

    def generate_trajectory(self, image, user_query, topology=None, state=None,
                            previous_movement=None, mov_history=None, test=False):
        
        if test:
            return {
                "room": "hallway",
                "movement": "B3",
                "state": "Oriented Towards Door",
                "description": "Test mode enabled",
                "door_position": "not_visible"
            }

        img_b64_str = self.encode_image_base64(image)
        
        # Montamos los inputs igual que en GPT
        user_inputs = [
            {"type": "text", "text": user_query},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64_str}"}}
        ]

        if topology:
            user_inputs.insert(1, {"type": "text", "text": "Topology: " + json.dumps(topology)})
            
        if state:
            user_inputs.insert(1, {"type": "text", "text": "Current FSM state: " + state})
            
        if previous_movement:
            user_inputs.insert(1, {"type": "text", "text": "Previous movement command: " + previous_movement})
            
        # Cargamos los prompts desde tus .txt
        system_prompt = load_system_prompt(
            path_dir="prompts",
            system_prompt_path="drone_system_prompt.txt",
            output_prompt_path="output_prompt.txt",
            curr_state=state
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_inputs}
        ]        
        
        print("Current FSM state: " + state)

        try:
            # Llamada a la API local de Ollama
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"}, # Forzamos salida JSON
                temperature=self.temperature
            )
            
            response_json = response.choices[0].message.content.strip()
            return json.loads(response_json)

        except Exception as e:
            logger.exception(f"Error generando trayectoria con Ollama: {str(e)}")
            return {}