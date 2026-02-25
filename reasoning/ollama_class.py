import os
import json
import logging
import base64
from openai import OpenAI
from reasoning.load_prompts import load_system_prompt
from reasoning.VLMModel import VLMBaseModel

logger = logging.getLogger(__name__)

# Añadimos la función para limpiar el markdown si el modelo responde con ```json
def parse_json(json_output: str):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output.strip()

class OllamaDescriptor(VLMBaseModel):
    def __init__(self, model="qwen3-vl:8b", temperature=0.0):
        # Configuramos el cliente para apuntar a Ollama en el propio servidor
        self.client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama'
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
        
        print("Enviando petición a Ollama (Qwen3-VL)...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
                # ELIMINADO intencionadamente: response_format={"type": "json_object"}
            )
            
            # Extraemos el texto en crudo
            response_text = response.choices[0].message.content.strip()
            
            print(f"Respuesta recibida. Limpiando JSON...")
            
            # Limpiamos el markdown por si acaso y cargamos el JSON
            clean_json = parse_json(response_text)
            return json.loads(clean_json)

        except Exception as e:
            logger.exception(f"Error generando trayectoria con Ollama: {str(e)}")
            return {}