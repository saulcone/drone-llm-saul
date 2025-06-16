import random
import numpy as np
import cv2
import requests
import json
import uuid
import platform
import argparse
import time

from io import BytesIO
from pynput import keyboard
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)

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

# --- Unity Side Channels ---
class IntSenderChannel(SideChannel):
    def __init__(self):
        # Must match the GUID used in Unity!
        super().__init__(uuid.UUID("12345678-1234-1234-1234-1234567890ab"))

    def on_message_received(self, msg: IncomingMessage) -> None:
        pass  # Unity-to-Python messages not needed

    def send_int(self, value: int) -> None:
        msg = OutgoingMessage()
        msg.write_int32(value)
        self.queue_message_to_send(msg)

# --- Funciones principales ---
def read_obs(env, behavior_drone):

    decision_steps, _ = env.get_steps(behavior_drone)
    obs = decision_steps.obs

    img = obs[0][0]
    real_img = img * 255.0
    height = obs[1][0][0]
    pos = obs[2][0][:3]
    yaw = obs[2][0][3]
    collision_state = obs[2][0][4]

    return img, real_img, height, pos, yaw, collision_state

def show_observations(img, height, pos, yaw, state):

    print(f"Height: {height}")
    print(f"Position: {pos}")
    print(f"Yaw: {yaw}")
    print(f"State: {state}")

    cv2.imshow("Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

def manual_control(key_ctrl):

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

def move_drone(cont, movement, env=None, action=None):

    for i in range(cont):
        pred = np.array([[movement[0], movement[1], movement[2], movement[3], 0]], dtype=np.float32)
        action.add_continuous(pred)
        env.set_actions(behavior_drone, action)
        env.step()

def move_drone_fast(cont, movement, env=None, action=None):

    pred = np.array([[movement[0]*cont, movement[1]*cont, movement[2]*cont, movement[3]*cont, 0]], dtype=np.float32)
    action.add_continuous(pred)
    env.set_actions(behavior_drone, action)
    env.step()

def run_simulation(args, topology):

    key_ctrl = KeyController()
    int_channel = IntSenderChannel()

    manual = args.manual
    seed = random.seed(args.seed)

    if platform.system() == "Windows":
        env = UnityEnvironment(file_name="./build/WAF_environment", seed=seed, side_channels=[int_channel], no_graphics=False)
    else:
        env = UnityEnvironment(file_name="./build_linux/WAF_environment", seed=seed, side_channels=[int_channel], no_graphics=False)

    env.reset()
    int_channel.send_int(args.spawn_id)
    spec = env.behavior_specs[list(env.behavior_specs)[0]]
    behavior_drone = list(env.behavior_specs)[0]

    state = "Recognize Room"
    query = args.query

    prev_movement = ""
    mov_history = []

    while True:

        mov_history.append(prev_movement)
        if len(mov_history) > 10:
            mov_history.pop(0)

        decision_steps, terminal_steps = env.get_steps(behavior_drone)
        action = spec.action_spec.random_action(len(decision_steps))

        if key_ctrl.is_pressed('o'):
            manual = True
        if key_ctrl.is_pressed('p'):
            manual = False
        if key_ctrl.is_pressed('q'):
            break
        if key_ctrl.is_pressed('r'):
            env.reset()

        if manual:
            movement = manual_control(key_ctrl)
            pred = np.array([[*movement, 0]], dtype=np.float32)
            action.add_continuous(pred)
            env.set_actions(behavior_drone, action)
            env.step()

            img, real_img, height, pos, yaw, collision_state = read_obs(env, behavior_drone)
            show_observations(img, height, pos, yaw, collision_state)

        else:

            img, real_img, height, pos, yaw, collision_state = read_obs(env, behavior_drone)
            show_observations((img, height, pos, yaw, collision_state))

            response = get_cmd_vel(real_img, query, topology, state, prev_movement, mov_history)
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

# --- Main ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Drone Control Simulator")
    parser.add_argument('--manual', action='store_true', help='Enable manual control mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--spawn_id', type=int, default=1, help='Spawn point for the drone')
    parser.add_argument('--n_exec', type=int, default=1, help='Number of executions to run')
    parser.add_argument('--addr', type=str, default="http://172.18.33.110:5001", help='Address of the server')
    parser.add_argument('--query', type=str, default="Go to the bedroom and take a photo of the mirror.", help='Query for the drone')
    args = parser.parse_args()

    topology = {"living room": ["bedroom", "bathroom"], "bedroom": ["living room"], "bathroom": ["living room"]}

    run_simulation(args, topology)

    
