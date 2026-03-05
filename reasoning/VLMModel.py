from abc import ABC, abstractmethod
import base64
import cv2


class VLMBaseModel(ABC):
    @abstractmethod
    def generate_trajectory(self, image: cv2.Mat, 
                            user_query: str, 
                            topology: str, 
                            state: str,
                            previous_movement: str,
                            mov_history: str,
                            test: bool) -> str:
        pass

    def encode_image_base64(self, image):
        """Function to encode the image as base64"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode("utf-8") 
    
        
class ReasoningModel:
    @staticmethod
    def create_model(model: str, **kwargs) -> VLMBaseModel:
        if "gpt" in model:
            from reasoning.gpt_class import GPTDescriptor
            return GPTDescriptor(model, temperature=kwargs["temperature"],
                                 max_tokens=kwargs["max_tokens"],
                                 top_p=kwargs["top_p"], img_type="image/jpeg")
        elif "o4" in model:
            from reasoning.o_models_class import OModelDescriptor
            return OModelDescriptor(model, max_tokens=kwargs["max_tokens"],
                                    reasoning_effort=kwargs["reasoning_effort"])
        elif "gemini" in model:
            from reasoning.gemini import GeminiDescriptor
            return GeminiDescriptor(model, temperature=kwargs["temperature"])
        
        elif "qwen" in model.lower() and "cloud" in model.lower():
            from reasoning.gpt_class import GPTDescriptor
            # Usamos la clase GPT porque se conecta por API web
            return GPTDescriptor(model, temperature=kwargs.get("temperature", 0.0),
                                 max_tokens=kwargs.get("max_tokens", 300),
                                 top_p=kwargs.get("top_p", 0.2))
        
        else:
            raise ValueError(f"Model not implemented: {model}")