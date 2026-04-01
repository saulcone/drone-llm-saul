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
        self.model = model
        self.temperature = temperature

    def generate_trajectory(self, image, user_query, topology=None,
                            mission_phase=None, mission_description=None, target_description=None,
                            previous_movement=None, mov_history=None, agent_memory=None, test=False):
        
        if test:
            return {"room": "hallway", "movement": "A2", "reasoning": "Test", "mission_progress": "Testing"}

        # 1. Encode image to Base64
        img_b64_str = self.encode_image_base64(image)
        
        # 2. Build user context
        user_text_parts = [user_query]
        
        if topology: 
            user_text_parts.append("Topology: " + json.dumps(topology))
        if previous_movement: 
            user_text_parts.append("Previous movement command: " + previous_movement)
        if mov_history: 
            user_text_parts.append("Movement history (last commands): " + str(mov_history))
        
        # Episodic memory injection
        if agent_memory: 
            user_text_parts.append("Agent internal memory (Notes from previous step): " + str(agent_memory))
            
        user_text = "\n\n".join(user_text_parts)

        # 3. Load unified prompt with mission injection
        system_prompt = load_system_prompt(
            path_dir="prompts",
            mission_phase=mission_phase or "GENERAL",
            mission_description=mission_description or "Navigate freely",
            target_description=target_description or "Unknown"
        )

        # 4. Build Ollama messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text, "images": [img_b64_str]}
        ]        

        try:
            # 5. Call Ollama API
            response = ollama.chat(model=self.model, messages=messages, format='json', options={"temperature": self.temperature})
            
            # 6. Safe JSON parsing
            response_text = response['message']['content'].strip()
            
            # Clean markdown tags that Qwen sometimes returns
            if response_text.startswith("```json"): 
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"): 
                response_text = response_text[3:-3].strip()

            return json.loads(response_text)
            
        except Exception as e:
            logger.exception(f"Error calling Ollama model {self.model}: {str(e)}")
            return None