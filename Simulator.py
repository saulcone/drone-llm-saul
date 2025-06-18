import random
import numpy as np
import cv2
import requests
import json
import uuid
import platform
import argparse
import time
import math
import os

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

    real_img = cv2.resize(img, (640, 480)) * 255.0  # Convert to 0-255 range
    real_img = np.clip(real_img, 0, 255).astype(np.uint8)

    height = obs[1][0][0]
    pos = obs[2][0][:3]
    yaw = obs[2][0][3]
    collision_state = obs[2][0][4]

    cv2.imshow("Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

    return img, real_img, height, pos, yaw, collision_state

def show_observations(img, height, pos, yaw, state):

    os.system('cls' if platform.system() == 'Windows' else 'clear')
    print(f"Height: {height}")
    print(f"Position: {pos}")
    print(f"Yaw: {yaw}")
    print(f"State: {state}")

def show_state(state, room, pred_room, goal_state, query, description, door_position):

    os.system('cls' if platform.system() == 'Windows' else 'clear')
    print(f"State: {state}")
    print(f"Room: {room}")
    print(f"Predicted Room: {pred_room}")
    print(f"Goal State: {goal_state}")
    print(f"Query: {query}")
    print(f"Description: {description}")
    print(f"Door Position: {door_position}")

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

def move_drone(cont, movement, env=None, action=None, behavior_drone=None):

    for i in range(cont):
        pred = np.array([[movement[0], movement[1], movement[2], movement[3], 0]], dtype=np.float32)
        action.add_continuous(pred)
        env.set_actions(behavior_drone, action)
        env.step()

def move_drone_fast(cont, movement, env=None, action=None, behavior_drone=None):

    pred = np.array([[movement[0]*cont/2, movement[1]*cont/2, movement[2]*cont/2, movement[3]*cont/2, 0]], dtype=np.float32)
    action.add_continuous(pred)
    env.set_actions(behavior_drone, action)
    env.step()

    pred = np.array([[movement[0]*cont/2, movement[1]*cont/2, movement[2]*cont/2, movement[3]*cont/2, 0]], dtype=np.float32)
    action.add_continuous(pred)
    env.set_actions(behavior_drone, action)
    env.step()

# --- Score sistem ---
def calculate_room(pos):
    x, y, z = pos

    # Bedroom (Cuarto): [(−1, −3.25), (4, −9.25)]
    if -1 <= x <= 4 and -9.25 <= z <= -3.25:
        return "bedroom"

    # Bathroom (Baño): [(−5.45, −9.2), (−1.05, −4.8)]
    elif -5.45 <= x <= -1.05 and -9.2 <= z <= -4.8:
        return "bathroom"

    # Living room (Salón): unión de dos zonas
    # Parte 1: [(−5.45, −3.15), (−1, −4.7)]
    if -5.45 <= x <= -1 and -4.7 <= z <= -3.15:
        return "living room"
    # Parte 2: [(−5.45, −3.15), (2.75, 5.35)]
    elif -5.45 <= x <= 2.75 and -3.15 <= z <= 5.35:
        return "living room"

    return None

def is_close_and_facing(x1, z1, x2, z2, yaw, dist=1.5, angle=25):

    close = False
    facing = False

    # Distancia euclidiana
    dx = x2 - x1
    dz = z2 - z1
    distance = math.sqrt(dx**2 + dz**2)

    if distance < dist:
        close = True

    # Ángulo entre los puntos en grados
    angle_to_target = math.degrees(math.atan2(dx, dz)) % 360

    # Diferencia angular mínima entre yaw y dirección (también en [0, 360))
    angle_diff = abs((yaw - angle_to_target + 180) % 360 - 180)

    if angle_diff <= angle or angle_diff >= (360-angle):
        facing = True

    return close, facing

def calculate_goal(goal_id, pos, yaw, room=None, pred_room=None):

    goal_rooms = ["bathroom", "bedroom", "living room"]
    # Reach room goals
    if goal_id < len(goal_rooms):
        if room == goal_rooms[goal_id] and pred_room == room:
            return True
        else:
            return False

    # Umbrales de distancia y ángulo
    dist_threshold = 2.0  # Distancia mínima para considerar "cerca"
    angle_threshold = 25  # Ángulo mínimo para considerar "de frente"

    # Diccionario de objetivos con coordenadas
    goals = {
        3: ("tv", (2.5, 1.2), "living room"),             # "Find the TV in the living room"
        4: ("refrigerator", (-4.5, 3.65), "living room"), # "Find the refrigerator in the living room"
        5: ("tap", (-5.2, -1.1), "living room"),          # "Find the faucet in the living room"
        6: ("mirror", (3.75, -5.65), "bedroom"),      # "Find the mirror in the bedroom"
        7: ("cabinet", (3.25, -4.0), "bedroom"),      # "Find the cabinet in the bedroom"
        8: ("sink", (-3.95, -5.25), "bathroom"),       # "Find the sink in the bathroom"
        9: ("toilet", (-5.25, -6.85), "bathroom"),     # "Find the toilet in the bathroom"
    }

    if goal_id not in goals:
        return False  # ID inválido

    _, (gx, gz), _ = goals[goal_id]  # Ajuste de índice para el diccionario
    x, y, z = pos

    close, facing = is_close_and_facing(x, z, gx, gz, yaw, dist=dist_threshold, angle=angle_threshold)

    return close and facing

def get_query(query_id):
    queries = [
        "Go to the bathroom",
        "Go to the bedroom",
        "Go to the living room",
        "Find the TV in the living room",
        "Find the refrigerator in the living room",
        "Find the faucet in the living room",
        "Find the mirror in the bedroom",
        "Find the cabinet in the bedroom",
        "Find the sink in the bathroom",
        "Find the toilet in the bathroom"
    ]
    if 0 <= query_id < len(queries):
        return queries[query_id]
    else:
        return ""

# --- Main Simulation Function ---
def run_simulation(args, topology, json_data=None):

    key_ctrl = KeyController()
    int_channel = IntSenderChannel()

    manual = args.manual

    if platform.system() == "Windows":
        env = UnityEnvironment(file_name="./build/WAF_environment", side_channels=[int_channel], no_graphics=False)
    else:
        env = UnityEnvironment(file_name="./build_linux/WAF_environment", side_channels=[int_channel], no_graphics=False)

    env.reset()

    spec = env.behavior_specs[list(env.behavior_specs)[0]]
    behavior_drone = list(env.behavior_specs)[0]

    int_channel.send_int(args.spawn_id)
    decision_steps, _ = env.get_steps(behavior_drone)
    action = spec.action_spec.random_action(len(decision_steps))
    move_drone(10, [0, 0, 0, 0], env=env, action=action, behavior_drone=behavior_drone)

    state = "Recognize Room"
    query_id = args.query
    query = get_query(query_id)
    prev_movement = ""
    mov_history = []
    pred_room = ""
    command = ""
    description = ""
    door_position = ""
    total_steps = 0
    max_steps = args.max_steps  # Maximum steps to prevent infinite loop
    result = None

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

    # Video writer
    if args.save_video:
        frame_size = (640, 480)  # ajusta según tus imágenes
        fps = 24 if args.manual else 1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # o 'XVID' para AVI
        video_writer = cv2.VideoWriter(args.output_video, fourcc, fps, frame_size)

    while True:

        total_steps += 1

        mov_history.append(prev_movement)
        if len(mov_history) > 10:
            mov_history.pop(0)

        decision_steps, _ = env.get_steps(behavior_drone)
        action = spec.action_spec.random_action(len(decision_steps))

        if key_ctrl.is_pressed('q'):
            break
        if key_ctrl.is_pressed('r'):
            env.reset()

        img, real_img, height, pos, yaw, collision_state = read_obs(env, behavior_drone)
        room = calculate_room(pos)
        goal_state = calculate_goal(query_id, pos, yaw, room, pred_room)

        if args.save_video:
            img_text = f"Step: {total_steps}"
            img_with_text = cv2.putText(real_img.copy(), img_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            video_writer.write(cv2.cvtColor(img_with_text, cv2.COLOR_RGB2BGR))

        if manual:

            # Manual control mode
            show_state(state, room, pred_room, goal_state, query, description, door_position)
            print(f"Query: {query}  | Room: {room} | Goal State: {goal_state}")

            # set predicted room
            pred_room = room 
            
            movement = manual_control(key_ctrl)
            pred = np.array([[*movement, 0]], dtype=np.float32)
            action.add_continuous(pred)
            env.set_actions(behavior_drone, action)
            env.step()

        else:

            # Get command velocity from the server
            response = get_cmd_vel(real_img, query, topology, state, prev_movement, mov_history)
            command = response.get("movement")
            state = response.get("state")
            pred_room = response.get("room", "")
            description = response.get("description", "")
            door_position = response.get("door_position", "")

            if args.fast:
                move_drone_fast(*move_map.get(command, (10, [0, 0, 0, 0])), env, action, behavior_drone)
            else:
                move_drone(*move_map.get(command, (10, [0, 0, 0, 0])), env, action, behavior_drone)

            prev_movement = command

        if args.save_logs:

            log_data = {
                "step": str(total_steps),
                "height": str(height),
                "position": str(pos),
                "yaw": str(yaw),
                "collision_state": str(collision_state),
                "command": command,
                "state": state,
                "room": room,
                "pred_room": pred_room,
                "description": description,
                "door_position": door_position,
                "goal_state": goal_state,
            }

            json_data["Step_" + str(total_steps)] = log_data

        if total_steps >= max_steps:
            print(f"Reached maximum steps: {max_steps}. Ending simulation.")
            result = "max_steps_reached"
            break
        
        if goal_state and (state == "Final" or args.manual):
            print("Goal reached! Ending simulation.")
            result = "goal_reached"
            break

        if collision_state == 0.0:
            print("Collision detected! Ending simulation.")
            result = "collision_detected"
            break

    if args.save_logs and json_data:

        end_info = {
            "total_steps": total_steps,
            "result": result
        }

        json_data["End_Info"] = end_info

        with open(args.output_file, "w") as info_file:
                json.dump(json_data, info_file, indent=2)
        print(f"Logs saved to {args.output_file}")

    if args.save_video:
        video_writer.release()
        print(f"Video saved to {args.output_video}")

    env.close()

def run_experiment(args, topology):

    spawn_range = range(1, 11) if args.spawn_id == -1 else [args.spawn_id]

    for spawn_id in spawn_range:  # Spawn points from 1 to 10
        for repeat in range(args.num_repeats):

            # Create output directory for the experiment
            output_dir = f"experiments/{args.experiment_name}"
            os.makedirs(output_dir, exist_ok=True)

            args.spawn_id = spawn_id
            args.output_file = f"{output_dir}/spawn_{spawn_id}_repeat_{repeat+1}.json" if args.save_logs else None
            args.output_video = f"{output_dir}/spawn_{spawn_id}_repeat_{repeat+1}.mp4" if args.save_video else None

            print(f"Experiment: {args.experiment_name} | Spawn: {spawn_id} | Repeat: {repeat+1}/{args.num_repeats}")

            # Save experiment info, query, and topology
            if args.save_logs:

                experiment_info = {
                    "experiment_name": args.experiment_name,
                    "query": get_query(args.query),
                    "query_id": args.query,
                    "topology": json.dumps(topology),
                    "spawn_id": spawn_id,
                }

                json_data = {"Experiment_Info": experiment_info}

            run_simulation(args, topology, json_data=json_data if args.save_logs else None)

# --- Main ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Drone Control Simulator")

    parser.add_argument('--fast', '-f', action='store_true', help='Enable fast movement mode')
    parser.add_argument('--save_logs', '-l', action='store_true', help='Enable logging of observations')
    parser.add_argument('--run_experiment', '-e', action='store_true', help='Run multiple executions of the simulation')
    parser.add_argument('--experiment_name', '-n', type=str, default="default_experiment", help='Name of the experiment for logging')
    parser.add_argument('--manual', '-m', action='store_true', help='Enable manual control mode')
    #parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--spawn_id', '-i', type=int, default=-1, help='Spawn point for the drone')
    parser.add_argument('--num_repeats', '-r', type=int, default=1, help='Number of repeats for each spawn point')
    parser.add_argument('--addr', '-a', type=str, default="http://172.18.33.110:5001", help='Address of the server')
    parser.add_argument('--query', '-q', type=int, default=0, help='Query id for the drone (0-9)')
    parser.add_argument('--save_video', '-v', action='store_true', help='Enable saving video of the simulation')
    parser.add_argument('--max_steps', '-s', type=int, default=50, help='Maximum steps to prevent infinite loop')

    args = parser.parse_args()

    topology = {"living room": ["bedroom", "bathroom"], "bedroom": ["living room"], "bathroom": ["living room"]}

    if args.run_experiment:
        run_experiment(args, topology)
        
    else:

        if args.save_logs:
            args.output_file = f"experiment_{args.experiment_name}_spawn_{args.spawn_id}.json"

            experiment_info = {
                "experiment_name": args.experiment_name,
                "query": get_query(args.query),
                "query_id": args.query,
                "topology": topology,
                "spawn_id": args.spawn_id,
            }

            json_data = {"Experiment_Info": experiment_info}
                    
        if args.save_video:
            args.output_video = f"experiment_{args.experiment_name}_spawn_{args.spawn_id}.mp4"

        run_simulation(args, topology, json_data=json_data if args.save_logs else None)

    
