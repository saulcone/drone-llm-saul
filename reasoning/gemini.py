import os
import base64
from google import genai
from google.genai import types
from dotenv import load_dotenv
import base64
import json
import cv2
from typing import Tuple

import json
import random
import io
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor

import dataclasses
import numpy as np
import base64

from reasoning.load_prompts import load_system_prompt

# Load the API key from the .env file
load_dotenv()


def parse_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output


class GeminiDescriptor:
    def __init__(self, model: str="gemini-2.5-pro-preview-06-05", 
                 temperature: float=0):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model
        self.temperature = temperature

        self.safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ]
        

    def generate_trajectory(self, image, user_query, topology=None, state=None,
                            previous_movement=None, mov_history=None, test=False):
        if test:
            return {
                "room": "hallway",
                "movement": "B3",
                "state": "Oriented Towards Door",
                "description": "Visible hallway with a door, move robot to face the door"
            }
        
        # Build the text content as a single string
        user_text_parts = [user_query]
        print(state)

        if state:
            user_text_parts.append("Current FSM state: " + state)
            
        if topology:
            user_text_parts.append("Topology: " + json.dumps(topology))
            
        if previous_movement:
            user_text_parts.append("Previous movement command: " + previous_movement)
        
        # Combine all text parts into a single string
        user_text = "\n\n".join(user_text_parts)
            
        system_prompt = load_system_prompt(
            path_dir="prompts",
            system_prompt_path="drone_system_prompt.txt",
            output_prompt_path="output_prompt.txt",
            curr_state=state
        )   
        
        print("Current FSM state: " + state)

        im = Image.fromarray(np.uint8(image))
        im.thumbnail([1024,1024], Image.Resampling.LANCZOS)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[user_text, im],
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                safety_settings=self.safety_settings,
            )
        )  
        res = json.loads(parse_json(response.text))
        print(res)

        return res