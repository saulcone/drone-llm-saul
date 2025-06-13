import random
import numpy as np
import cv2
import requests
import json
from io import BytesIO
from pynput import keyboard
from mlagents_envs.environment import UnityEnvironment

# --- Teclado (pynput) ---
class KeyController:
    def __init__(self):
        self.pressed_keys = set()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        try:
            self.pressed_keys.add(key.char)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            self.pressed_keys.discard(key.char)
        except AttributeError:
            pass

    def is_pressed(self, char):
        return char in self.pressed_keys

key_ctrl = KeyController()

# --- Funciones principales ---
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
    cv2.waitKey(1)

def manual_control():
    speed = 1.0
    rotSpeed = 1.0
    for_back_velocity = 0
    left_right_velocity = 0
    up_down_velocity = 0
    yaw_velocity = 0

    if key_ctrl.is_pressed('w'):
        for_back_velocity = speed
    elif key_ctrl.is_pressed('s'):
        for_back_velocity = -speed

    if key_ctrl.is_pressed('d'):
        left_right_velocity = speed
    elif key_ctrl.is_pressed('a'):
        left_right_velocity = -speed

    if key_ctrl.is_pressed('i'):
        up_down_velocity = speed
    elif key_ctrl.is_pressed('k'):
        up_down_velocity = -speed

    if key_ctrl.is_pressed('l'):
        yaw_velocity = rotSpeed
    elif key_ctrl.is_pressed('j'):
        yaw_velocity = -rotSpeed

    return left_right_velocity, for_back_velocity, yaw_velocity, up_down_velocity

def get_cmd_vel(img, query, topology, state, prev_movement, mov_history, addr = "http://172.18.33.110:5001"):
    url = addr + '/cmd_vel'
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = BytesIO(img_encoded.tobytes())
    files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
    data = {'query': query, 'topology': json.dumps(topology), 'state': state,
            'prev_movement': prev_movement, 'mov_history': json.dumps(mov_history)}

    try:
        response = requests.post(url, files=files, data=data)
        print(response.json())
        return response.json()
    except Exception as e:
        print("Error decoding response:", e)
        return {"movement": "", "state": "Start"}

def take_photo(env, behavior_name):
    decision_steps, _ = env.get_steps(behavior_name)
    obs = decision_steps.obs
    img = obs[0][0]
    img = cv2.resize(img, (640, 480))
    cv2.imshow("Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img * 255.0

def move_drone(cont, movement, env=None, action=None):
    for i in range(cont):
        pred = np.array([[movement[0], movement[1], movement[2], movement[3], 0]], dtype=np.float32)
        action.add_continuous(pred)
        env.set_actions(behavior_name, action)
        env.step()

# --- Main ---
if __name__ == "__main__":

    manual = True
    seed = random.seed(42)
    env = UnityEnvironment(file_name="./build_linux/WAF_environment", seed=seed, side_channels=[], no_graphics=False)
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
    query = "Go to the bedroom and take a photo of the mirror."
    prev_movement = ""
    topology = {"living room": ["bedroom", "bathroom"], "bedroom": ["living room"], "bathroom": ["living room"]}
    final_img = None
    mov_history = []

    while True:
        mov_history.append(prev_movement)
        if len(mov_history) > 10:
            mov_history.pop(0)

        decision_steps, terminal_steps = env.get_steps(behavior_name)
        action = spec.action_spec.random_action(len(decision_steps))
        obs = decision_steps.obs

        if key_ctrl.is_pressed('o'):
            manual = True
        if key_ctrl.is_pressed('p'):
            manual = False
        if key_ctrl.is_pressed('q'):
            break
        if key_ctrl.is_pressed('r'):
            env.reset()

        if manual:
            movement = manual_control()
            pred = np.array([[*movement, 0]], dtype=np.float32)
            action.add_continuous(pred)
            env.set_actions(behavior_name, action)
            env.step()
        else:
            img = final_img.copy() if final_img is not None else take_photo(env, behavior_name)
            final_img = None
            response = get_cmd_vel(img, query, topology, state, prev_movement, mov_history)
            command = response.get("movement")
            state = response.get("state")

            move_map = {
                "A1": (10, [0, 0.1, 0, 0]),
                "A2": (25, [0, 0.1, 0, 0]),
                "A3": (50, [0, 0.1, 0, 0]),
                "B1": (15, [0, 0, -0.1, 0]),
                "B2": (45, [0, 0, -0.1, 0]),
                "B3": (90, [0, 0, -0.1, 0]),
                "C1": (15, [0, 0, 0.1, 0]),
                "C2": (45, [0, 0, 0.1, 0]),
                "C3": (90, [0, 0, 0.1, 0]),
                "D1": (10, [-0.1, 0, 0, 0]),
                "D2": (10, [0.1, 0, 0, 0])
            }

            move_drone(*move_map.get(command, (10, [0, 0, 0, 0])), env, action)
            prev_movement = command

    env.close()
