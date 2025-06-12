import os
from openai import OpenAI
from dotenv import load_dotenv
import base64
import json
import cv2

# Load the API key from the .env file
load_dotenv()


class GPTDescriptor:
    def __init__(self, model="gpt-4.1", max_tokens=300, temperature=0,
                 top_p=0.2, prompt_path="drone_system_prompt.txt"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
        # Load system prompt from file
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
            raise RuntimeError(f"System prompt file not found at {prompt_path}")
        
        
    def encode_image_base64(self, image):
        """Function to encode the image as base64"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode("utf-8")    
    
    
    def generate_trajectory(self, image, user_query, topology=None, state=None,
                            previous_movement=None,
                            img_type="image/jpeg", test=False):
        """Generate a trajectory using the GPT model."""
        if test:
            return {
                "room": "hallway",
                "movement": "B3",
                "state": "Oriented Towards Door",
                "description": "Visible hallway with a door, move robot to face the door"
            }

        
        img_b64_str = self.encode_image_base64(image)
        
        user_inputs = [
            {"type": "text", "text": user_query},
            {"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"}}
        ]

        if topology:
            user_inputs.insert(1, {
                "type": "text",
                "text": "Topology: " + json.dumps(topology)
            })
            
        if state:
            user_inputs.insert(1, {
                "type": "text",
                "text": "Current FSM state: " + state
            })
            
        if previous_movement:
            user_inputs.insert(1, {
                "type": "text",
                "text": "Previous movement command: " + previous_movement
            })

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_inputs}
        ]        
        
        print("Current FSM state: " + state)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self.temperature, # controls randomness
                max_completion_tokens=self.max_tokens, # controls response length)
                top_p=self.top_p, # controls diversity, adjusts probability distribution
                frequency_penalty=0, # affect repetition
                presence_penalty=0, # affect repetition                
            )

            chatgpt_response_json = response.choices[0].message.content.strip()
            return json.loads(chatgpt_response_json)

        except Exception as e:
            print(f"An error occurred decoding chatgpt json response: {str(e)}")
