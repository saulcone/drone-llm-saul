import os
import json
import logging
import cv2
from pathlib import Path
from dotenv import load_dotenv
import ollama

# Load .env from reasoning folder
load_dotenv(Path(__file__).parent / ".env", override=True)

from reasoning.load_prompts import load_system_prompt
from reasoning.VLMModel import VLMBaseModel
from reasoning.waypoint_controller import WaypointController

logger = logging.getLogger(__name__)

class OllamaDescriptor(VLMBaseModel):
    def __init__(self, model="qwen3-vl:235b-cloud", temperature=0.0):
        self.model = model
        self.temperature = temperature
        # Initialize waypoint controller for SPF system
        self.waypoint_controller = WaypointController()

    def generate_trajectory(self, image, user_query, topology=None,
                            mission_phase=None, mission_description=None, target_description=None,
                            previous_movement=None, mov_history=None, agent_memory=None, test=False):
        
        if test:
            return {
                "room": "hallway", 
                "movement": "A2", 
                "reasoning": "Test",
                "target_point": [0.5, 0.45],
                "distance": "LEJOS",
                "confidence": "ALTA",
                "mission_progress": "Testing"
            }

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

            parsed_response = json.loads(response_text)
            
            # 7. SPF: Convert waypoint to movement command
            # Pass mission_phase to detect if we're in scanning mode
            is_scanning = mission_phase and "ESCANEO" in str(mission_phase).upper()
            parsed_response = self._process_waypoint_response(parsed_response, is_scanning=is_scanning)
            
            return parsed_response
            
        except Exception as e:
            logger.exception(f"Error calling Ollama model {self.model}: {str(e)}")
            return None
    
    def _process_waypoint_response(self, response, is_scanning=False):
        """
        Process VLM response with SPF (See, Point, Fly) system.
        
        Takes the target_point from VLM and converts it to a movement command
        using the WaypointController.
        
        Args:
            response: Dict with target_point, distance, confidence, etc.
            is_scanning: If True, be lenient with oscillation detection
            
        Returns:
            Dict with added 'movement' field computed from waypoint
        """
        # Extract waypoint data
        target_point = response.get("target_point", [0.5, 0.5])
        distance = response.get("distance", "DESCONOCIDO")
        confidence = response.get("confidence", "MEDIA")
        
        # Validate target_point format
        if not isinstance(target_point, (list, tuple)) or len(target_point) != 2:
            logger.warning(f"Invalid target_point format: {target_point}, using default center")
            target_point = [0.5, 0.5]
        
        # Convert waypoint to movement command
        movement = self.waypoint_controller.compute_command(
            target_point=target_point,
            distance=distance,
            confidence=confidence
        )
        
        # Check for oscillation and apply anti-oscillation if needed
        # Pass is_scanning to be lenient during scanning phase
        if self.waypoint_controller.check_oscillation(is_scanning=is_scanning):
            logger.info("[SPF] Oscillation detected, applying anti-oscillation strategy")
            movement = self.waypoint_controller.get_anti_oscillation_command(target_point, distance, is_scanning=is_scanning)
            response["_oscillation_override"] = True
        
        # Add computed movement to response
        response["movement"] = movement
        
        # Log SPF decision
        logger.info(f"[SPF] Point: {target_point}, Dist: {distance}, Conf: {confidence} -> {movement}")
        
        return response