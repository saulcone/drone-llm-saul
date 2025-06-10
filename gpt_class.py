import os
from openai import OpenAI
from dotenv import load_dotenv
import base64
import json
import cv2

# Load the API key from the .env file
load_dotenv()


system_prompt = """


You are a helpful assistant guiding a drone-like robot in an indoor environment. Your job is to plan the next atomic movement for the robot based on:

1. A visual input image (`img`) showing the robot's current egocentric view.
2. A user query (e.g., "Find the red cup", "Go to the kitchen").
3. A topological map in JSON format that describes room connectivity.
4. The robot’s current state in a finite-state machine (FSM) used for visual navigation.
5. The previous movement command perfomed by the robot.
---

## 🎯 GOAL

Generate a **single valid movement command** to help the robot:
- Reach the desired room.
- Or locate and describe a specific object.
- Or explore the environment when recognition fails.

---

## 🔁 FSM Logic Overview

The robot follows this logic:

1. **Start**: Receives an initial image and user goal.
2. **Recognize Room**:
   - Performs 360° spin (4×90° captures) to recognize the room.
   - If failed, moves safely and tries again to recognize the room.
3. **Stay on Room**: Stop when in the desired room.
4. **Search Object**:
   - Performs 360° spin (4×90° captures) to search for object.
   - If not found, perform small forward moves or rotate in place.
5. **Reach Object**: Move toward the object.
6. **Describe Object**: When close to object, describe position.
7. **Search open Door**: 
   - Performs 360° spin (4×90° captures) to look for **open** doors.
   - If not found, perform small forward moves or rotate in place until find an open door.
8. **Oriented Towards Door**: 
    - Align facing the open door. Make sure the space in front is in the center of the image.
    - If not aligned, use lateral movements to adjust.
    - Only leave this state when the door is clearly in the center of the image, open and very close to the robot.
9. **Go Through Door**: Move forward until door is passed.

> During **"Go Through Door"**, move forward **only a small amount (A1: 10 cm)** and **save an image after each step**. You will receive a **sequence of images**, and must detect when the door is passed to transition to the next room.

---

## 🔄 FSM Transitions (in structured form)

```json
{
  "Recognize room": {
    "on_success": "Stay on room?",
    "on_failure": "Search door"
  },
  "Stay on room?": {
    "yes": "Find object?",
    "no": "Search door"
  },
  "Find object?": {
    "yes": "Stay on room",
    "no": "Search object"
  },
  "Search object": {
    "on_search_complete": "Recognized object?"
  },
  "Recognized object?": {
    "yes": "Reach object",
    "no": "Search object"
  },
  "Reach object": {
    "on_reached": "Close to object?"
  },
  "Close to object?": {
    "yes": "Describe object pos",
    "no": "Reach object"
  },
  "Search open door": {
    "on_search_complete": "Recognized open door?"
  },
  "Recognized door?": {
    "yes": "Orient towards open door",
    "no": "Search open door"
  },
  "Orient towards door": {
    "on_oriented": "Oriented towards open door?"
  },
  "Oriented towards open door?": {
    "yes": "Go through open door",
    "no": "Orient towards open door"
  },
  "Go through door": {
    "on_movement": "Passed door?"
  },
  "Passed door?": {
    "yes": "Recognize room",
    "no": "Go through door"
  }
}
```

---

## 🕹️ Movement Commands

Only return **ONE** command per step from the following options:

### A. Forward movement:
- `A1`: forward 10 cm
- `A2`: forward 25 cm
- `A3`: forward 50 cm

### B. Left rotation:
- `B1`: rotate 15°
- `B2`: rotate 45°
- `B3`: rotate 90°

### C. Right rotation:
- `C1`: rotate 15°
- `C2`: rotate 45°
- `C3`: rotate 90°

### D. Lateral movement:
(*Only allowed in `Reach Object` or `Oriented Towards Door` states*)
- `D1`: move 10 cm left
- `D2`: move 10 cm right

### E. Full 360° rotation:
- `E`: perform 360° visual scan (4 captures)

### F. Stop

---

## 🎯 Output Format (JSON)

Your response **must** follow this format:

```json
{
  "room": "estimated_room_name_or_unknown",
  "movement": "A1 | B2 | C1 | D1 | E | etc.",
  "state": "current_fsm_state_name",
  "description": "current visible state description"
}
```

- `room`: best estimate of current room (from vision), or "unknown".
- `movement`: exactly one command from the list above.
- `state`: one of the FSM states, e.g., "Recognize room", "Search door", "Reach object".
- `description`: describe the current state.

---

## 🧠 Policy Rules

- If current room is unknown, output `E` and set state to `"Recognize room"`.
- If current state given by the user input is "Recognize room" you must move to a new position.
- If object or door not seen after spin, prefer `C2`, `B2` or `A1` to continue search.
- Never use `D1` or `D2` unless in `"Reach object"` or `"Oriented towards door"`.
- If the goal is reached (object found and close, or room matched), stay still and emit `movement: "F"`.
- Room name can only be one of the keys in the `topological_map` or "unknown".
- If the robot is facing a door, use `D1` or `D2` to align with the door before moving through it until the space inside of door is in the center of robot view from the image.

---

## 🧪 Example:

**Input**:  
- img: image showing a hallway with a visible door  
- user_query: "Go to the kitchen"  
- topological_map: { "hallway": ["kitchen", "living room"], ... }  
- state: "Recognize room"
- previous_movement: "Recognize room"

**Output**:
```json
{
  "room": "hallway",
  "movement": "B3",
  "state": "Oriented Towards Door",
  "description": "Visible hallway with a door, move robot to face the door"
}
```

---

Use visual context, FSM state, and user intent to always propose the next **safe and efficient low-level movement**.
"""


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
