import os
from openai import OpenAI
from dotenv import load_dotenv
import base64
import json
import cv2

# Load the API key from the .env file
load_dotenv()


def load_system_prompt(system_prompt_path, output_prompt_path, curr_state):
    """Load the system prompt from a file."""
    print("current state: " + curr_state)
    if curr_state == "Recognize Room":
        state_prompt_path = "state_0.txt"
    # elif curr_state == "Position in the center of room":
    #     state_prompt_path = "state_1.txt"
    elif curr_state == "Stay on Room":
        state_prompt_path = "state_2.txt"
    elif curr_state == "Search Object":
        state_prompt_path = "state_3.txt"
    elif curr_state == "Reach Object":
        state_prompt_path = "state_4.txt"
    elif curr_state == "Describe Object":
        state_prompt_path = "state_5.txt"
    elif curr_state == "Search open Door":
        state_prompt_path = "state_6.txt"
    elif curr_state == "Oriented Towards Door":
        state_prompt_path = "state_7.txt"
    elif curr_state == "Go Through Door":
        state_prompt_path = "state_8.txt"
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt =  f.read()
        with open(state_prompt_path, "r", encoding="utf-8") as f:
            state_prompt =  f.read()
        with open(output_prompt_path, "r", encoding="utf-8") as f:
            output_prompt =  f.read()        
        prompt = system_prompt + "\n" + state_prompt + "\n" + output_prompt
        return prompt
    except FileNotFoundError:
        raise RuntimeError(f"System prompt file not found at {prompt}")
    

class GPTDescriptor:
    def __init__(self, model="gpt-4.1", max_tokens=300, temperature=0,
                 top_p=0.2):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
        
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
            
        system_prompt = load_system_prompt(
            "drone_system_prompt.txt",
            "output_prompt.txt",
            state
        )

        messages = [
            {"role": "system", "content": system_prompt},
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
