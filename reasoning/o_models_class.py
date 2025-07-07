import os
from openai import OpenAI
from dotenv import load_dotenv
import base64
import json
import cv2
import logging

from reasoning.load_prompts import load_system_prompt
from reasoning.VLMModel import VLMBaseModel

logger = logging.getLogger(__name__)

# Load the API key from the .env file
load_dotenv()
    

class OModelDescriptor(VLMBaseModel):
    def __init__(self, model="o4-mini", max_tokens=300,
                 reasoning_effort= "medium", img_type="image/jpeg"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.img_type=img_type
        

    def generate_trajectory(self, image, user_query, topology=None, state=None,
                            previous_movement=None,
                            mov_history=None,
                            test=False):
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
            {"type": "image_url", "image_url": {"url": f"data:{self.img_type};base64,{img_b64_str}"}}
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                max_completion_tokens=self.max_tokens, # controls response length) 
            )
            logger.debug("completion_tokens: %s", response.usage.completion_tokens)
            logger.debug("reasoning_tokens: %s", response.usage.completion_tokens_details.reasoning_tokens)

            chatgpt_response_json = response.choices[0].message.content.strip()
            return json.loads(chatgpt_response_json)

        except Exception:
            logger.exception("An error occurred while generating trajectory with OpenAI API")


# ChatCompletion(id='chatcmpl-BqOTkFCB1WaKMDqQrYzsdrcSzbyi2',
#                choices=[Choice(finish_reason='stop', index=0, 
#                                logprobs=None, 
#                                message=ChatCompletionMessage(content='{\n  "room": "bedroom",\n  "movement": "B2",\n  "state": "Search Object",\n  "description": "Bed with headboard, window with curtain, nightstand and lamp; no mirror visible",\n  "door_position": "not_visible"\n}', 
#                                                              refusal=None, role='assistant', annotations=[], audio=None, function_call=None, tool_calls=None))], 
#                created=1751826852, model='o4-mini-2025-04-16', object='chat.completion', service_tier='default', 
#                system_fingerprint=None, 
#                usage=CompletionUsage(completion_tokens=396, prompt_tokens=1286, total_tokens=1682, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=320, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0)))