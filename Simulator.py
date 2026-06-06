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
from reasoning.wall_proximity_detector import WallProximityDetector, SafetyEnvelope
# Adaptive Distance Scaling for SPF system (usado en modo scanning dentro del continuous controller)
from reasoning.adaptive_scaling import AdaptiveScaler
# Phase 4+5 Enhanced: Continuous proportional control with safety envelope v2
from reasoning.continuous_controller import ContinuousController, SafetyEnvelopeV2, ControlMode, VelocityCommand


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

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
        response = requests.post(url, files=files, data=data, timeout=120)
        print(response.json())
        return response.json()
    except requests.exceptions.Timeout:
        print("[ERROR] VLM request timed out after 120s")
        return {"movement": "E", "reasoning": "Timeout", "mission_progress": "Error"}
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
    if 0 <= goal_id < len(goal_rooms):
        if room == goal_rooms[goal_id] and pred_room == room:
            return True
        else:
            return False

    # Umbrales de distancia y ángulo
    dist_threshold = 3.0  # Distancia mínima para considerar "cerca"
    angle_threshold = 25  # Ángulo mínimo para considerar "de frente"

    # Diccionario de objetivos con coordenadas
    goals = {
        3: ("tv", (2.5, 1.2), "living room & kitchen"),             
        4: ("refrigerator", (-4.5, 3.65), "living room & kitchen"), 
        5: ("tap", (-5.2, -1.1), "living room & kitchen"),          
        6: ("mirror", (3.75, -5.65), "bedroom"),      
        7: ("cabinet", (3.25, -4.0), "bedroom"),      
        8: ("sink", (-3.95, -5.25), "bathroom"),       
        9: ("toilet", (-5.25, -6.85), "bathroom"),     
    }

    if goal_id not in goals:
        return False  # ID inválido

    # NUEVO: Extraemos también la habitación objetivo de la tupla
    _, (gx, gz), target_room = goals[goal_id]  
    
    # CRÍTICO: Si el dron no está físicamente en la misma habitación que el objeto, 
    # es imposible que lo haya alcanzado (evita ganar a través de paredes)
    if room != target_room:
        return False

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

    # Derive target from query_id so MissionManager matches the VLM's assigned query
    _query_to_mission = {
        0: ("sink", "bathroom"),
        1: ("bed", "bedroom"),
        2: ("refrigerator", "living room & kitchen"),
        3: ("tv", "living room & kitchen"),
        4: ("refrigerator", "living room & kitchen"),
        5: ("faucet", "living room & kitchen"),
        6: ("mirror", "bedroom"),
        7: ("cabinet", "bedroom"),
        8: ("sink", "bathroom"),
        9: ("toilet", "bathroom"),
    }
    _qid = getattr(args, 'query', -1)
    _default_obj = getattr(args, 'target_object', 'sink')
    _default_room = getattr(args, 'target_room', 'bathroom')
    target_object, target_room = _query_to_mission.get(_qid, (_default_obj, _default_room))
    
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
    crossing_lock_streak = 0
    max_crossing_lock_streak = 2
    crossing_commit_frames = 0
    CROSSING_COMMIT_FRAMES = 4
    CROSSING_COMMIT_STEPS = 14
    
    # Safety Guard: detector de proximidad sin VLM (rápido y fiable)
    wall_detector = WallProximityDetector(edge_threshold=50, imbalance_threshold=0.25)
    safety_envelope = SafetyEnvelope()

    safety_override = None
    use_safety_envelope = getattr(args, 'safety', True)

    # Adaptive Distance Scaling (usado en modo scanning dentro del continuous controller)
    adaptive_scaler = AdaptiveScaler()

    # Continuous proportional controller with safety envelope v2
    continuous_controller = ContinuousController()
    safety_envelope_v2 = SafetyEnvelopeV2()
    use_continuous = getattr(args, 'continuous', False)

    # Video writer
    if args.save_video:
        frame_size = (480, 360)  # ajusta según tus imágenes
        frame_size_2 = (1280, 720)  # ajusta según tus imágenes de la cámara trasera
        fps = 1
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

        # Get mission context from mission manager
            mission_ctx = mission_manager.get_mission_context()
            
            # === STATE LATCH: Inject recovery context if needed ===
            recovery_context = mission_manager.get_recovery_context()
            if recovery_context:
                # Prepend recovery context to agent memory
                print(f"[STATE LATCH] Injecting recovery context into agent memory")
                agent_memory = recovery_context + "\n\n" + agent_memory
            
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
            
            # === STATE LATCH: Apply room correction ===
            # If latch is active, override VLM's room prediction
            if mission_manager.room_latch_active:
                corrected_room = mission_manager.apply_state_latch(pred_room)
                if corrected_room != pred_room:
                    pred_room = corrected_room
                    # Also update response for logging
                    response["room"] = corrected_room
            
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

            # === SAFETY ENVELOPE (Phase 4) ===
            # Virtual safety layer that overrides VLM when collision risk detected
            original_command = command
            safety_override = None
            
            if use_safety_envelope:
                # Check if current phase involves crossing doors
                current_phase = mission_manager.current_phase
                is_crossing = current_phase and "TRAVES" in str(current_phase).upper()
                is_approaching = current_phase and "APROXIM" in str(current_phase).upper()
                
                # Also check mission_progress for crossing indicators
                crossing_progress = any(x in mission_progress.lower() for x in 
                    ["crossing", "cruzando", "atraves", "travesía", "entering", "entrando"])
                
                if is_crossing or is_approaching or crossing_progress:
                    # Activate safety envelope during door crossing/approaching
                    safe_command, override_reason, analysis = safety_envelope.check_crossing_safety(
                        image_cv2=real_img,
                        target_point=target_point,
                        distance=distance,
                        intended_command=command
                    )
                    
                    if override_reason:
                        # Safety envelope intervened
                        # BLOQUEADO: Ya no dejamos que la Fase 4 cambie el comando discreto
                        # command = safe_command 
                        safety_override = override_reason

                        # Send feedback to VLM via memory update (suppressed during TRAVESIA to avoid confusing door frames with walls)
                        _phase = mission_manager.current_phase if 'mission_manager' in dir() else None
                        if not (_phase and "TRAVES" in str(_phase).upper()):
                            agent_memory += f" [SAFETY ALERT: {override_reason}]"
                    elif analysis and "imbalance" in analysis:
                        # Log safety status even when not intervening
                        print(f"[SAFETY] ✓ Clear (imbalance={analysis['imbalance']:.2f})")
            
            # === END SAFETY GUARDS ===

            # === CONTINUOUS CONTROLLER (Phase 4+5 Enhanced) ===
            # Proportional velocity control with integrated safety envelope
            # HYBRID MODE: Use discrete commands for scanning, continuous for approach/crossing
            if use_continuous and command != "E":
                # Determine mode first
                current_phase = mission_manager.current_phase
                ctrl_mode = continuous_controller.determine_mode(
                    tuple(target_point), distance, mission_progress, command,
                    str(current_phase) if current_phase else "",
                    reasoning,
                    agent_memory
                )
                continuous_controller.set_mode(ctrl_mode)
                
                # For SCANNING mode, use discrete commands (faster rotation)
                if ctrl_mode == ControlMode.SCANNING:
                    # Fall through to adaptive scaler for fast discrete rotation
                    print(f"[CONTINUOUS] Mode: scanning -> using discrete commands for speed")
                    move_steps, move_vec = adaptive_scaler.get_adaptive_move(command, distance, confidence)
                    scaling_phase = "HYBRID-scanning"
                    # More descriptive output based on command type
                    if command.startswith("B") or command.startswith("C"):
                        direction = "LEFT" if command.startswith("B") else "RIGHT"
                        print(f"[HYBRID] Discrete ROTATE {direction} ~{move_steps}°")
                else:
                    # For APPROACHING, CROSSING, CENTERING - use continuous control
                    velocity_cmd = continuous_controller.compute_velocity(
                        target_point=tuple(target_point),
                        distance=distance,
                        confidence=confidence
                    )
                    
                    # === CROSSING LOCK: Mute Safety-V2 only in strict crossing conditions ===
                    current_phase = mission_manager.current_phase
                    is_travesia_phase = current_phase and "TRAVES" in str(current_phase).upper()
                    crossing_progress = any(word in mission_progress.lower() for word in
                        ["crossing door", "crossing", "cruzando", "atraves", "travesía", "entering"])
                    blind_crossing_hint = any(word in f"{mission_progress.lower()} {reasoning.lower()} {agent_memory.lower()}" for word in
                        ["blind crossing", "blind sprint", "empujón final", "empujon final", "efecto túnel", "efecto tunel", "final push", "excepción del 30%", "excepcion del 30%"])
                    x = target_point[0] if isinstance(target_point, (list, tuple)) and target_point else 0.5
                    is_centered_crossing = distance == "MUY_CERCA" and abs(x - 0.5) <= 0.05
                    crossing_lock_enabled = (
                        ctrl_mode == ControlMode.CROSSING and
                        command.startswith("A") and
                        is_travesia_phase and
                        (crossing_progress or blind_crossing_hint) and
                        is_centered_crossing
                    )
                    if crossing_lock_enabled and crossing_commit_frames == 0 and room != target_room:
                        crossing_commit_frames = CROSSING_COMMIT_FRAMES
                        print(f"[CROSSING COMMIT] Hard forward commit armed ({crossing_commit_frames} frames)")
                    
                    hard_cross_active = crossing_commit_frames > 0
                    if hard_cross_active:
                        if room == target_room:
                            print(f"[CROSSING COMMIT] Room reached — cancelling commit early")
                            crossing_commit_frames = 0
                            hard_cross_active = False
                        else:
                            crossing_lock_enabled = True
                            crossing_commit_frames -= 1

                    if crossing_lock_enabled and room != target_room and not hard_cross_active:
                        crossing_lock_streak += 1
                    else:
                        crossing_lock_streak = 0

                    if crossing_lock_enabled and not hard_cross_active and crossing_lock_streak > max_crossing_lock_streak:
                        print("[CROSSING LOCK] ⚠️ Crossing budget exceeded without room change. Re-enabling Safety-V2.")
                        crossing_lock_enabled = False
                        # Degrade to cautious forward to avoid repeated blind impacts.
                        velocity_cmd.vx = min(velocity_cmd.vx, 0.12)

                    # Also skip Safety-V2 during CENTERING in TRAVESÍA: door frame triggers
                    # imbalance=1.0 which collapses vx→0.016 and makes strafe dominant → oscillation
                    centering_travesia_skip = is_travesia_phase and ctrl_mode == ControlMode.CENTERING

                    if crossing_lock_enabled:
                        if hard_cross_active:
                            print(f"[CROSSING LOCK] Hard commit active: forcing forward ({crossing_commit_frames} frames left)")
                        else:
                            print("[CROSSING LOCK] Safety Envelope V2 MUTED for blind sprint!")
                        # Avance balístico con pequeña corrección lateral si la puerta aparece off-center.
                        # vy=0 era la causa de colisiones cuando el commit disparaba con X transitorio de 0.5
                        # pero el drone estaba físicamente descentrado (puerta real en X=0.65-0.75).
                        door_x = target_point[0] if isinstance(target_point, (list, tuple)) and len(target_point) > 0 else 0.5
                        x_err = door_x - 0.5
                        # Corrección solo cuando error > 0.10 (evita micro-oscilaciones en puertas bien centradas)
                        lateral_correction = float(np.clip(x_err * 0.4, -0.08, 0.08)) if abs(x_err) > 0.10 else 0.0
                        velocity_cmd.vx = 0.25
                        velocity_cmd.vy = lateral_correction
                        velocity_cmd.vyaw = 0.0

                        # NO aplicamos el safety_envelope_v2
                        safety_analysis = None
                    elif centering_travesia_skip:
                        # CENTERING during TRAVESÍA: Safety-V2 confuses door frame for wall → skip
                        print(f"[SAFETY-V2] ⏭️ Skipped (CENTERING in TRAVESÍA — door frame false positive)")
                        safety_analysis = None
                    else:
                        # Apply safety envelope v2 (ABSOLUTE AUTHORITY for other modes)
                        # Para TODO lo demás (Scanning, Approaching, Centering), la seguridad ACTÚA
                        safe_velocity, safety_reason, safety_analysis = safety_envelope_v2.apply_safety_envelope(
                            image_cv2=real_img,
                            velocity_cmd=velocity_cmd,
                            distance=distance
                        )
                        
                        if safety_reason:
                            # Safety envelope intervened
                            print(f"[SAFETY-V2] ⚠️ {safety_reason}")
                            velocity_cmd = safe_velocity
                            safety_override = safety_reason
                        else:
                            imbalance = safety_analysis.get("imbalance", 0) if safety_analysis else 0
                            print(f"[SAFETY-V2] ✓ Clear (imbalance={imbalance:.2f})")
                    
                    # Convert velocity to movement parameters
                    move_steps, move_vec = continuous_controller.velocity_to_steps(velocity_cmd)
                    if hard_cross_active and move_steps < CROSSING_COMMIT_STEPS:
                        move_steps = CROSSING_COMMIT_STEPS
                    
                    # Log continuous controller status
                    mode = continuous_controller.mode.value
                    print(f"[CONTINUOUS] Mode: {mode}, Velocity: vx={velocity_cmd.vx:.3f} vy={velocity_cmd.vy:.3f} vyaw={velocity_cmd.vyaw:.3f}")
                    print(f"[CONTINUOUS] Output: steps={move_steps}, vec={[f'{v:.2f}' for v in move_vec]}")
                    scaling_phase = f"CONTINUOUS-{mode}"
            

            if command != "E":  # Don't move on wait command
                if args.fast:
                    move_drone_fast(move_steps, move_vec, env, action, behavior_drone)
                else:
                    move_drone(move_steps, move_vec, env, action, behavior_drone)
            else:
                # === SOLUCIÓN TIMEOUT: Forzar a Unity a avanzar en el tiempo ===
                # Si el comando es "E" (espera) y no ejecutamos move_drone,
                # Unity se congela y Python satura la red pidiendo inferencias sin parar.
                # Al enviar un vector de [0,0,0,0], el dron hace 'hover' y el tiempo corre.
                move_drone(10, [0.0, 0.0, 0.0, 0.0], env, action, behavior_drone)

            prev_movement = command

    

        if args.save_logs:
            continuous_status = continuous_controller.get_status()
            safety_v2_status = safety_envelope_v2.get_status()

            log_data = {
                "step": str(total_steps),
                "position": str(pos),
                "yaw": str(yaw),
                "collision_state": str(collision_state),
                "command": command,
                "original_command": original_command,
                "safety_override": safety_override,
                "target_point": target_point,
                "distance": distance,
                "confidence": confidence,
                "oscillation_override": bool(oscillation_override),
                "scaling_phase": scaling_phase,
                "move_steps": int(move_steps) if move_steps is not None else None,
                "move_vec": [float(v) for v in move_vec] if move_vec is not None else None,
                "continuous_enabled": bool(use_continuous),
                "continuous_status": continuous_status,
                "safety_v2_status": safety_v2_status,
                "mission_phase": mission_manager.get_phase_name(),
                "room": room,
                "pred_room": pred_room,
                "reasoning": reasoning,
                "mission_progress": mission_progress,
                "agent_memory": agent_memory,
                "goal_state": bool(goal_state) if goal_state is not None else None,
                "elapsed_time": float(elapsed_time),
            }

            json_data["Step_" + str(total_steps)] = log_data

            with open(args.output_file, "w") as info_file:
                json.dump(json_data, info_file, indent=2, cls=NumpyEncoder)

        if total_steps >= max_steps:
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
                json.dump(json_data, info_file, indent=2, cls=NumpyEncoder)
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
    parser.add_argument('--spawn_id', '-i', type=int, default=-1, help='Spawn point for the drone (1-10)')
    parser.add_argument('--num_repeats', '-r', type=int, default=1, help='Number of repeats for each spawn point')
    parser.add_argument('--addr', '-a', type=str, default="http://localhost:5001", help='Address of the server')
    parser.add_argument('--query', '-q', type=int, default=-1, help='Query id for the drone (0-9)')
    parser.add_argument('--save_video', '-v', action='store_true', help='Enable saving video of the simulation')
    parser.add_argument('--max_steps', '-s', type=int, default=10, help='Maximum steps to prevent infinite loop')
    parser.add_argument('--safety', action='store_true', default=True, help='Enable safety envelope for door crossing (default: on)')
    parser.add_argument('--no-safety', dest='safety', action='store_false', help='Disable safety envelope')
    parser.add_argument('--continuous', action='store_true', default=False, help='Enable continuous proportional control')

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

    
