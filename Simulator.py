import random
import numpy as np
import keyboard 
import cv2
import requests
import json

from io import BytesIO
from mlagents_envs.environment import UnityEnvironment

def show_observations(obs):

    img = obs[0][0]
    height = obs[1][0][0]
    pos = obs[2][0][:3]
    state = obs[2][0][3]

    print(f"Image shape: {img.shape}")
    print(f"Height: {height}")
    print(f"Position: {pos}")
    print(f"State: {state}")

    cv2.imshow("Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)  # Display the image for a brief moment

def manual_control():

    speed = 1.0
    rotSpeed = 1.0

    for_back_velocity = 0
    left_right_velocity = 0
    up_down_velocity = 0
    yaw_velocity = 0
        
    if keyboard.is_pressed('w'):
        for_back_velocity = speed
    elif keyboard.is_pressed('s'):
        for_back_velocity = -speed
        
    if keyboard.is_pressed('d'):
        left_right_velocity = speed
    elif keyboard.is_pressed('a'):
        left_right_velocity = -speed
        
    if keyboard.is_pressed('i'):
        up_down_velocity = speed
    elif keyboard.is_pressed('k'):
        up_down_velocity = -speed
        
    if keyboard.is_pressed('l'):
        yaw_velocity = rotSpeed
    elif keyboard.is_pressed('j'):
        yaw_velocity = -rotSpeed
        
    return left_right_velocity, for_back_velocity, yaw_velocity, up_down_velocity

def get_cmd_vel(img, query, topology, state, prev_movement, addr = "http://172.18.33.110:5001"):

    url = addr + '/cmd_vel'

    # Encode image to memory
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = BytesIO(img_encoded.tobytes())

    files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}

    data = {'query': query,
            'topology': json.dumps(topology), 
            'state': state, 
            'prev_movement': prev_movement}
    try:
        response = requests.post(url, files=files, data=data)
        print(response.json())
        return response.json()

    except Exception as e:
        print("Error decoding response:", e)
        return {"movement": "", "state": "Start"}  # Default action if request fails

def take_photo(env, behavior_name):

    decision_steps, _ = env.get_steps(behavior_name)
    obs = decision_steps.obs

    img = obs[0][0]
    img = cv2.resize(img, (640, 480))  
    cv2.imshow("Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1) 
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img*255.0

def move_drone(cont, movement, env=None, action=None):

    left_right_velocity = movement[0]
    for_back_velocity = movement[1]
    yaw_velocity = movement[2]
    up_down_velocity = movement[3]

    for i in range(cont):
        
        pred = np.array([[left_right_velocity, for_back_velocity, yaw_velocity, up_down_velocity, 0]], dtype=np.float32)
        #print(f"Action: {pred}")
        
        action.add_continuous(pred)
        env.set_actions(behavior_name, action)
        env.step()

if __name__ == "__main__":

    manual = True   

    seed = random.seed(42)
    env = UnityEnvironment(file_name="./build/WAF_environment", seed=seed, side_channels=[], no_graphics=False)
    env.reset()

    behavior_name = list(env.behavior_specs)[0]
    print(f"Name of the behavior : {behavior_name}")
    spec = env.behavior_specs[behavior_name]
    print("Number of observations : ", len(spec.observation_specs))

    if spec.action_spec.continuous_size > 0:
        print(f"There are {spec.action_spec.continuous_size} continuous actions")
    if spec.action_spec.is_discrete():
        print(f"There are {spec.action_spec.discrete_size} discrete actions")

    state = "Recognize Room"

    query = "Go to the bedhroom and take a photo of the mirror."

    prev_movement = ""

    topology = {
        "living room": ["bedroom", "bathroom"],
        "bedroom": ["living room"],
        "bathroom": ["living room"]
    }

    final_img = None

    mov_history = []

    while True:
    
        mov_history.append(prev_movement)
        if len(mov_history) > 10:
            mov_history.pop(0)

        decision_steps, terminal_steps = env.get_steps(behavior_name)
        action = spec.action_spec.random_action(len(decision_steps))

        obs = decision_steps.obs

        imgs = []
            
        if keyboard.is_pressed('o'):
            manual = True
        if keyboard.is_pressed('p'):
            manual = False
            
        if keyboard.is_pressed('q'):
            break
        if keyboard.is_pressed('r'):
            env.reset()
            
        if manual:

            left_right_velocity, for_back_velocity, yaw_velocity, up_down_velocity = manual_control()

            pred = np.array([[left_right_velocity, for_back_velocity, yaw_velocity, up_down_velocity, 0]], dtype=np.float32)
            
            action.add_continuous(pred)
            env.set_actions(behavior_name, action)
            env.step()

        else:

            if final_img is None:
                img = take_photo(env, behavior_name)
            else:
                img = final_img.copy()
            
            final_img = None

            response = get_cmd_vel(img, query, topology, state, prev_movement, mov_history)
            
            command = response.get("movement")
            state = response.get("state")
                    
            if command == "A1":
                move_drone(10, [0, 0.1, 0, 0], env, action)
            elif command == "A2":
                move_drone(25, [0, 0.1, 0, 0], env, action)
            elif command == "A3":
                move_drone(50, [0, 0.1, 0, 0], env, action)
            elif command == "B1":
                move_drone(15, [0, 0, -0.1, 0], env, action)
            elif command == "B2":
                move_drone(45, [0, 0, -0.1, 0], env, action)
            elif command == "B3":
                move_drone(90, [0, 0, -0.1, 0], env, action)
            elif command == "C1":
                move_drone(15, [0, 0, 0.1, 0], env, action)
            elif command == "C2":
                move_drone(45, [0, 0, 0.1, 0], env, action)
            elif command == "C3":
                move_drone(90, [0, 0, 0.1, 0], env, action)
            elif command == "D1":
                move_drone(10, [-0.1, 0, 0, 0], env, action)
            elif command == "D2":
                move_drone(10, [0.1, 0, 0, 0], env, action)
            else:
                move_drone(10, [0, 0, 0, 0], env, action)

            prev_movement = command        

    env.close()