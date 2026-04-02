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

# Safety Guard: Detector de proximidad a paredes (sin VLM)
from reasoning.wall_proximity_detector import WallProximityDetector
# Adaptive Distance Scaling for SPF system
from reasoning.adaptive_scaling import AdaptiveScaler, STATIC_MOVE_MAP
# MPC Controller for smooth trajectory planning
from reasoning.mpc_controller import MPCController, create_mpc_controller

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

def read_obs(env, behavior_drone):
    decision_steps, _ = env.get_steps(behavior_drone)
    obs = decision_steps.obs

    # Sensor 0: Siempre es la Cámara Frontal
    img = obs[0][0]
    real_img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (480, 360)) * 255.0
    real_img = np.clip(real_img, 0, 255).astype(np.uint8)

    # Lógica adaptativa según el número de sensores recibidos
    if len(obs) == 3:
        # CASO ACTUAL: Falta la cámara trasera
        # Creamos una imagen negra vacía para que el resto del código no falle
        real_img_2 = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Los sensores se han movido un puesto hacia arriba
        height = obs[1][0][0]       # Altura ahora es obs[1]
        pos_data = obs[2][0]        # Posición ahora es obs[2]

    else:
        # CASO ORIGINAL (4 sensores): Frontal, Trasera, Altura, Posición
        img_2 = obs[1][0]
        real_img_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR) * 255.0
        real_img_2 = np.clip(real_img_2, 0, 255).astype(np.uint8)

        height = obs[2][0][0]
        pos_data = obs[3][0]

    # Extraer datos comunes del vector de posición
    pos = pos_data[:3]
    yaw = pos_data[3]
    collision_state = pos_data[4]

    cv2.imshow("Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

    return img, real_img, real_img_2, height, pos, yaw, collision_state

def show_observations(img, height, pos, yaw, state):

    os.system('cls' if platform.system() == 'Windows' else 'clear')
    print(f"Height: {height}")
    print(f"Position: {pos}")
    print(f"Yaw: {yaw}")
    print(f"State: {state}")

def show_state(mission_phase, room, pred_room, goal_state, query, reasoning, mission_progress):
    """Display current mission state for manual control mode"""
    os.system('cls' if platform.system() == 'Windows' else 'clear')
    print(f"Mission Phase: {mission_phase}")
    print(f"Room: {room}")
    print(f"Predicted Room: {pred_room}")
    print(f"Goal State: {goal_state}")
    print(f"Query: {query}")
    print(f"Reasoning: {reasoning}")
    print(f"Mission Progress: {mission_progress}")

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

def get_cmd_vel(img, query, topology, mission_phase, mission_description, target_description, 
                current_room, prev_movement, mov_history, agent_memory="", addr = "http://localhost:5001"):
    """
    Send image and mission context to server, get movement command.
    
    Args:
        img: Current camera image
        query: User query/goal
        topology: Room connectivity graph
        mission_phase: Current mission phase (e.g., "BÚSQUEDA GLOBAL")
        mission_description: Detailed mission description
        target_description: Target object/location
        current_room: Current room (ground truth from simulator)
        prev_movement: Previous movement command
        mov_history: History of recent movements
        agent_memory: Episodic memory from previous step
        addr: Server address
        
    Returns:
        Dict with "reasoning", "movement", "room", "mission_progress", "memory_update"
    """
    url = addr + '/cmd_vel'
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = BytesIO(img_encoded.tobytes())
    files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
    data = {
        'query': query,
        'topology': json.dumps(topology),
        'mission_phase': mission_phase,
        'mission_description': mission_description,
        'target_description': target_description,
        'current_room': current_room,
        'prev_movement': prev_movement,
        'mov_history': json.dumps(mov_history),
        'agent_memory': agent_memory
    }

    try:
        response = requests.post(url, files=files, data=data)
        print(response.json())
        return response.json()
    except Exception as e:
        print("Error decoding response:", e)
        return {"movement": "E", "reasoning": "Error", "mission_progress": "Error"}

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
        return "living room & kitchen"
    # Parte 2: [(−5.45, −3.15), (2.75, 5.35)]
    elif -5.45 <= x <= 2.75 and -3.15 <= z <= 5.35:
        return "living room & kitchen"

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

    if distance < 0.5:
        angle = angle * 3

    elif distance < 1:
        angle = angle * 2

    # Ángulo entre los puntos en grados
    angle_to_target = math.degrees(math.atan2(dx, dz)) % 360

    # Diferencia angular mínima entre yaw y dirección (también en [0, 360))
    angle_diff = abs((yaw - angle_to_target + 180) % 360 - 180)

    if angle_diff <= angle or angle_diff >= (360-angle):
        facing = True

    return close, facing

def calculate_goal(goal_id, pos, yaw, room=None, pred_room=None):

    goal_rooms = ["bathroom", "bedroom", "living room & kitchen"]
    # Reach room goals
    if goal_id < len(goal_rooms):
        if room == goal_rooms[goal_id] and pred_room == room:
            return True
        else:
            return False

    # Umbrales de distancia y ángulo
    dist_threshold = 3.0  # Distancia mínima para considerar "cerca"
    angle_threshold = 25  # Ángulo mínimo para considerar "de frente"

    # Diccionario de objetivos con coordenadas
    goals = {
        3: ("tv", (2.5, 1.2), "living room & kitchen"),             # "Find the TV in the living room"
        4: ("refrigerator", (-4.5, 3.65), "living room & kitchen"), # "Find the refrigerator in the living room"
        5: ("tap", (-5.2, -1.1), "living room & kitchen"),          # "Find the faucet in the living room"
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
        "Go to the living room or kitchen",
        "Find the TV in the living room",
        "Find the refrigerator in the living room or kitchen",
        "Find the faucet in the living room or kitchen",
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
    move_drone(20, [0, 0, 0, 0], env, action, behavior_drone)

    # Mission setup
    from reasoning.mission_manager import MissionManager, MissionPhase
    
    # Parse target from query (e.g., query 8 = sink in bathroom)
    target_object = args.target_object if hasattr(args, 'target_object') else "sink"
    target_room = args.target_room if hasattr(args, 'target_room') else "bathroom"
    
    mission_manager = MissionManager(target_object=target_object, target_room=target_room)
    print(f"\n{'='*60}")
    print(f"[MISSION] Initialized: {target_object} in {target_room}")
    print(f"[MISSION] Starting phase: {mission_manager.get_phase_name()}")
    print(f"{'='*60}\n")
    
    query_id = args.query
    query = get_query(query_id)
    prev_movement = ""
    mov_history = []
    pred_room = ""
    command = ""
    reasoning = ""
    mission_progress = ""
    agent_memory = ""  # Episodic memory: the drone's own notes from previous steps
    total_steps = 0
    total_time = 0.0
    elapsed_time = 0.0
    max_steps = args.max_steps  # Maximum steps to prevent infinite loop
    result = None
    
    # Safety Guard: detector de proximidad sin VLM (rápido y fiable)
    wall_detector = WallProximityDetector(edge_threshold=50, imbalance_threshold=0.25)
    safety_override = None  # Guarda info cuando Safety Guard interviene
    
    # Adaptive Distance Scaling: scales movement based on distance to target
    adaptive_scaler = AdaptiveScaler()
    use_adaptive_scaling = getattr(args, 'adaptive', True)  # Enable by default
    
    # MPC Controller: smooth trajectory planning (Phase 4)
    mpc_controller = create_mpc_controller()
    use_mpc = getattr(args, 'mpc', False)  # Disabled by default, enable with --mpc

    # Static move_map for fallback/manual mode
    move_map = STATIC_MOVE_MAP

    # Video writer
    if args.save_video:
        frame_size = (480, 360)  # ajusta según tus imágenes
        frame_size_2 = (1280, 720)  # ajusta según tus imágenes de la cámara trasera
        fps = 24 if args.manual else 1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # o 'XVID' para AVI
        video_writer = cv2.VideoWriter(args.output_video, fourcc, fps, frame_size)
        video_writer_2 = cv2.VideoWriter(args.output_video.replace('.mp4', '_back.mp4'), fourcc, fps, frame_size_2)

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

        img, real_img, real_img_2, height, pos, yaw, collision_state = read_obs(env, behavior_drone)
        room = calculate_room(pos)
        goal_state = calculate_goal(query_id, pos, yaw, room, pred_room)

        if args.save_video:
            img_text = f"Step: {total_steps}"
            img_with_text = cv2.putText(real_img.copy(), img_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            video_writer.write(img_with_text)
            img_with_text_2 = cv2.putText(real_img_2.copy(), img_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            video_writer_2.write(img_with_text_2)

        if manual:

            # Manual control mode
            show_state(
                mission_manager.get_phase_name(),
                room, pred_room, goal_state, query,
                reasoning, mission_progress
            )
            print(f"Query: {query}  | Room: {room} | Goal State: {goal_state}")

            # set predicted room
            pred_room = room 
            
            # Initialize SPF variables for manual mode
            target_point = None
            distance = None
            confidence = None
            oscillation_override = False
            move_steps = None
            scaling_phase = "Manual"
            
            movement = manual_control(key_ctrl)
            pred = np.array([[*movement, 0]], dtype=np.float32)
            action.add_continuous(pred)
            env.set_actions(behavior_drone, action)
            env.step()

        else:

            # Get mission context from mission manager
            mission_ctx = mission_manager.get_mission_context()
            
            # Get command velocity from the server with mission context
            start_time = time.time()
            response = get_cmd_vel(
                real_img, query, topology,
                mission_phase=mission_ctx["phase"],
                mission_description=mission_ctx["description"],
                target_description=mission_ctx["target"],
                current_room=room,  # Ground truth room from simulator
                prev_movement=prev_movement,
                mov_history=mov_history,
                agent_memory=agent_memory,
                addr=args.addr
            )
            
            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            if response is None:
                print("Warning: Model returned None, skipping step.")
                continue

            # SPF System: Extract waypoint data and computed movement
            command = response.get("movement", "E")  # Already computed by waypoint_controller
            target_point = response.get("target_point", [0.5, 0.5])
            distance = response.get("distance", "DESCONOCIDO")
            confidence = response.get("confidence", "MEDIA")
            pred_room = response.get("room", "")
            reasoning = response.get("reasoning", "")
            mission_progress = response.get("mission_progress", "")
            agent_memory = response.get("memory_update", agent_memory)
            oscillation_override = response.get("_oscillation_override", False)
            
            # Log SPF decision
            print(f"[SPF] Point: {target_point}, Dist: {distance}, Conf: {confidence} -> {command}")
            if oscillation_override:
                print(f"[SPF] Oscillation detected, using anti-oscillation strategy")
            
            # Update mission phase based on VLM response
            mission_manager.update_phase(response, room)
            
            # Check if mission complete
            if mission_manager.is_complete():
                print(f"\n{'='*60}")
                print(f"🎉 MISIÓN COMPLETADA 🎉")
                print(f"Target: {target_object}")
                print(f"Steps: {total_steps}")
                print(f"Total time: {total_time:.2f}s")
                print(f"{'='*60}\n")
                result = "Success"
                break

            # === SAFETY GUARD (Disabled) ===
            # NOTE: Can be enabled for SPF system - no longer depends on FSM states
            # Use: wall_detector.get_safe_command(real_img, command)
            original_command = command
            safety_override = None
            # === END SAFETY GUARD ===

            # === MPC CONTROLLER (Phase 4) ===
            # Smooth continuous trajectory planning from waypoints
            if use_mpc and command != "E":
                # Determine if we're in scanning mode
                current_phase = mission_manager.get_current_phase()
                is_scanning = current_phase and "ESCANEO" in str(current_phase).upper()
                
                # Compute smooth velocity command
                velocity_cmd = mpc_controller.compute_velocity(
                    target_point=tuple(target_point),
                    distance=distance,
                    confidence=confidence,
                    is_scanning=is_scanning
                )
                
                # Convert to movement parameters
                move_steps, move_vec = mpc_controller.get_movement_steps(velocity_cmd)
                
                # Log MPC status
                status = mpc_controller.get_status()
                print(f"[MPC] Mode: {status['mode']}, Cmd: vx={status['last_cmd']['vx']:.2f} vy={status['last_cmd']['vy']:.2f} vyaw={status['last_cmd']['vyaw']:.2f}")
                scaling_phase = f"MPC-{status['mode']}"
                
            # === ADAPTIVE DISTANCE SCALING (legacy fallback) ===
            # Scale movement based on distance to target
            elif use_adaptive_scaling and command != "E":
                move_steps, move_vec = adaptive_scaler.get_adaptive_move(command, distance, confidence)
                scaling_phase = adaptive_scaler.get_phase_description()
                # More descriptive output based on command type
                if command.startswith("B") or command.startswith("C"):
                    direction = "LEFT" if command.startswith("B") else "RIGHT"
                    print(f"[ADAPTIVE] {scaling_phase}: ROTATE {direction} ~{move_steps}°")
                elif command.startswith("A"):
                    print(f"[ADAPTIVE] {scaling_phase}: FORWARD ~{move_steps}cm")
                else:
                    print(f"[ADAPTIVE] {scaling_phase}: {command} -> steps={move_steps}")
            else:
                move_steps, move_vec = move_map.get(command, (10, [0, 0, 0, 0]))
                scaling_phase = "Static"
            # === END MOVEMENT SCALING ===

            if command != "E":  # Don't move on wait command
                if args.fast:
                    move_drone_fast(move_steps, move_vec, env, action, behavior_drone)
                else:
                    move_drone(move_steps, move_vec, env, action, behavior_drone)

            prev_movement = command

    

        if args.save_logs:
            # Get MPC status if enabled
            mpc_status = mpc_controller.get_status() if use_mpc else None
            
            log_data = {
                "step": str(total_steps),
                "position": str(pos),
                "yaw": str(yaw),
                "collision_state": str(collision_state),
                "command": command,
                "original_command": original_command if not manual else command,
                "safety_override": safety_override if not manual else None,
                # SPF waypoint data
                "target_point": target_point if not manual else None,
                "distance": distance if not manual else None,
                "confidence": confidence if not manual else None,
                "oscillation_override": oscillation_override if not manual else False,
                # Adaptive scaling data
                "scaling_phase": scaling_phase if not manual else "Manual",
                "move_steps": move_steps if not manual else None,
                "move_vec": move_vec if not manual else None,
                # MPC data (Phase 4)
                "mpc_enabled": use_mpc,
                "mpc_status": mpc_status,
                # Mission data
                "mission_phase": mission_manager.get_phase_name() if not manual else "Manual",
                "room": room,
                "pred_room": pred_room,
                "reasoning": reasoning if not manual else "",
                "mission_progress": mission_progress if not manual else "",
                "agent_memory": agent_memory,
                "goal_state": goal_state,
                "elapsed_time": elapsed_time,
            }

            json_data["Step_" + str(total_steps)] = log_data

            with open(args.output_file, "w") as info_file:
                json.dump(json_data, info_file, indent=2)

        if total_steps >= max_steps and not args.manual:
            print(f"Reached maximum steps: {max_steps}. Ending simulation.")
            result = "max_steps_reached"
            break
        
        if goal_state:
            print("Goal reached! Ending simulation.")
            result = "goal_reached"
            break

        if collision_state == 0.0:
            print("Collision detected! Ending simulation.")
            result = "collision_detected"
            break
            break

    if args.save_logs and json_data:

        end_info = {
            "total_steps": total_steps,
            "total_time": total_time,
            "result": result
        }

        json_data["End_Info"] = end_info

        with open(args.output_file, "w") as info_file:
                json.dump(json_data, info_file, indent=2)
        print(f"Logs saved to {args.output_file}")

    if args.save_video:
        video_writer.release()
        video_writer_2.release()
        print(f"Video saved to {args.output_video}")
        print(f"Video (back) saved to {args.output_video.replace('.mp4', '_back.mp4')}")

    env.close()

def run_experiment(args, topology):

    #spawn_range = range(1, 11) if args.spawn_id == -1 else [args.spawn_id]
    spawn_range = [4, 8, 10] if args.spawn_id == -1 else [args.spawn_id]
    query_range = [0, 1, 4] if args.query == -1 else [args.query]


    for spawn_id in spawn_range:  # Spawn points from 1 to 10
        for query in query_range:  # Queries from 0 to 9
            for repeat in range(args.num_repeats):

                # Create output directory for the experiment
                output_dir = f"experiments/{args.experiment_name}"
                os.makedirs(output_dir, exist_ok=True)

                args.spawn_id = spawn_id
                args.query = query
                args.output_file = f"{output_dir}/spawn_{spawn_id}_query_{query}_repeat_{repeat+1}.json" if args.save_logs else None
                args.output_video = f"{output_dir}/spawn_{spawn_id}_query_{query}_repeat_{repeat+1}.mp4" if args.save_video else None

                print(f"Experiment: {args.experiment_name} | Spawn: {spawn_id} | Query: {query} | Repeat: {repeat+1}")

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
    parser.add_argument('--spawn_id', '-i', type=int, default=-1, help='Spawn point for the drone (1-10)')
    parser.add_argument('--num_repeats', '-r', type=int, default=1, help='Number of repeats for each spawn point')
    # parser.add_argument('--addr', '-a', type=str, default="http://172.18.33.110:5001", help='Address of the server')
    parser.add_argument('--addr', '-a', type=str, default="http://localhost:5001", help='Address of the server')
    parser.add_argument('--query', '-q', type=int, default=-1, help='Query id for the drone (0-9)')
    parser.add_argument('--save_video', '-v', action='store_true', help='Enable saving video of the simulation')
    parser.add_argument('--max_steps', '-s', type=int, default=10, help='Maximum steps to prevent infinite loop')
    
    # NEW: Mission parameters
    parser.add_argument('--target_object', '-to', type=str, default="sink", help='Target object to reach (e.g., sink, red cup)')
    parser.add_argument('--target_room', '-tr', type=str, default="bathroom", help='Target room where object is located')
    
    # NEW: Adaptive scaling
    parser.add_argument('--adaptive', action='store_true', default=True, help='Enable adaptive distance scaling (default: True)')
    parser.add_argument('--no-adaptive', dest='adaptive', action='store_false', help='Disable adaptive distance scaling')
    
    # NEW: MPC Controller (Phase 4)
    parser.add_argument('--mpc', action='store_true', default=False, help='Enable MPC smooth trajectory controller')
    parser.add_argument('--no-mpc', dest='mpc', action='store_false', help='Disable MPC (use discrete commands)')

    args = parser.parse_args()

    topology = {"living room & kitchen": ["bedroom", "bathroom"], "bedroom": ["living room & kitchen"], "bathroom": ["living room & kitchen"]}

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

    
