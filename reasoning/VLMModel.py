from abc import ABC, abstractmethod
import base64
import cv2


class VLMBaseModel(ABC):
    @abstractmethod
    def generate_trajectory(self, image: cv2.Mat, 
                            user_query: str, 
                            topology: str = None,
                            mission_phase: str = None,
                            mission_description: str = None,
                            target_description: str = None,
                            previous_movement: str = None,
                            mov_history: str = None,
                            agent_memory: str = None,
                            test: bool = False) -> dict:
        """
        Generate trajectory based on image and mission context.
        
        SPF System (See-Point-Fly):
        - VLM returns target_point [x,y] coordinates (0-1 normalized)
        - WaypointController converts to movement command
        
        Returns:
            dict with target_point, distance, confidence, room, reasoning, movement
        """
        pass

    def encode_image_base64(self, image):
        """Function to encode the image as base64"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode("utf-8") 
    
        
class ReasoningModel:
    """
    Factory for VLM models.
    
    Currently only supports Qwen3-VL cloud model for SPF system.
    Legacy FSM models (GPT, Gemini, o4) have been deprecated.
    """
    
    @staticmethod
    def create_model(model: str, **kwargs) -> VLMBaseModel:
        if "qwen" in model.lower() and "cloud" in model.lower():
            from reasoning.ollama_class import OllamaDescriptor
            return OllamaDescriptor(model, temperature=kwargs.get("temperature", 0.0))
        else:
            raise ValueError(
                f"Model not supported: {model}. "
                f"SPF system only supports 'qwen3-vl:235b-cloud' or similar Qwen cloud models."
            )