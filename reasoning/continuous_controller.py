"""
Proportional velocity controller for the SPF system.
Maps target_point [x,y] from VLM to continuous velocities [vx, vy, vyaw].
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ControlMode(Enum):
    """Operating modes with different control parameters"""
    SCANNING = "scanning"       # Searching for target - prioritize rotation
    APPROACHING = "approaching" # Moving toward visible target
    CROSSING = "crossing"       # Crossing door - very precise
    CENTERING = "centering"     # Aligning with door before crossing


@dataclass
class VelocityCommand:
    """Continuous velocity command for the drone"""
    vx: float      # Forward velocity (m/s), positive = forward
    vy: float      # Lateral velocity (m/s), positive = right (strafe)
    vz: float      # Vertical velocity (m/s), positive = up (NOT USED)
    vyaw: float    # Yaw rate (rad/s), positive = clockwise (right)
    
    def to_move_vector(self) -> list:
        """
        Convert to Unity move vector format.
        
        Unity expects: [strafe_X, forward_Y, yaw, unused]
        - Index 0: Strafe (lateral) - positive = right
        - Index 1: Forward - positive = forward
        - Index 2: Yaw (rotation) - positive = right (clockwise)
        - Index 3: Not used (was height in some versions)
        
        Our VelocityCommand:
        - vx = forward velocity
        - vy = lateral (strafe) velocity  
        - vyaw = yaw rate
        """
        return [self.vy, self.vx, self.vyaw, 0.0]  # [strafe, forward, yaw, unused]
    
    def magnitude(self) -> float:
        """Total velocity magnitude"""
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)


class ContinuousController:
    """
    Maps target_point [x,y] to continuous velocity commands.
    Lateral gains are kept small relative to forward to avoid door-frame overshoot.
    """
    
    # === VELOCITY LIMITS (m/s and rad/s) ===
    MAX_FORWARD_VEL = 0.3      # Max forward speed (reduced)
    MAX_LATERAL_VEL = 0.2
    MAX_YAW_RATE = 0.5         # Max rotation rate (rad/s, ~30°/s)
    
    # === PROPORTIONAL GAINS ===
    
    KP_LATERAL = 0.6
    KP_YAW = 0.6        # Yaw rate gain
    KP_FORWARD = 0.4    # Forward velocity gain
    
    # === MODE-SPECIFIC PARAMETERS ===
    MODE_PARAMS = {
        ControlMode.SCANNING: {
            "forward_scale": 0.0,    # Don't advance while scanning
            "lateral_scale": 0.0,    # Don't strafe while scanning
            "yaw_scale": 1.5,        # Fast rotation for scanning
        },
        ControlMode.APPROACHING: {
            "forward_scale": 1.0,    # Normal forward
            "lateral_scale": 1.0,    # Moderate lateral correction
            "yaw_scale": 0.3,        # Moderate yaw correction
        },
        ControlMode.CROSSING: {
            "forward_scale": 0.8,    # Slow forward
            "lateral_scale": 0.4,    # Good lateral correction for door alignment
            "yaw_scale": 0.1,        # Minimal yaw (avoid rotation while crossing)
        },
        ControlMode.CENTERING: {
            "forward_scale": 0.5,    # Avance diagonal proporcional a la alineación
            "lateral_scale": 1.0,    # Fuerte corrección lateral
            "yaw_scale": 0.2,        # Giro suave para encarar la puerta perpendicularmente
        }
    }
    
    # Lateral corrections don't scale with distance — we want small incremental steps regardless
    DISTANCE_SCALES = {
        "MUY_LEJOS": 0.8,     
        "LEJOS": 0.6,         
        "CERCA": 0.5,        
        "MUY_CERCA": 0.4,     
        "DESCONOCIDO": 0.4    
    }
    
    # === LATERAL MOVEMENT LIMITS ===
    # Maximum lateral movement per step in meters
    MAX_LATERAL_STEP_M = 0.15  # 15cm max per step - prevents overshooting
    
    def __init__(self):
        self.mode = ControlMode.APPROACHING
        self.last_cmd = VelocityCommand(0, 0, 0, 0)
        self.smoothing_factor = 0.0
        

        self.close_memory = 0
        self.was_recently_close = False  
        self.lost_frames = 0            
        
    def set_mode(self, mode: ControlMode):
        """Set operating mode"""
        self.mode = mode
        
    def determine_mode(self, target_point: Tuple[float, float], 
                       distance: str, mission_progress: str = "",
                       intended_command: str = "",
                       mission_phase: str = "",
                       reasoning_text: str = "",
                       memory_text: str = "") -> ControlMode:
        """Automatically determine the best control mode based on context."""
        x, y = target_point
        x_dev = abs(x - 0.5)
        intended = (intended_command or "").upper()
        phase = (mission_phase or "").upper()
        is_travesia_phase = "TRAVES" in phase

        progress = (mission_progress or "").lower()
        is_object_progress = any(word in progress for word in [
            "searching object", "approaching object", "reached target", "object", "sink"
        ])
        is_crossing_progress = any(word in progress for word in [
            "crossing door", "crossing", "cruzando", "atraves", "travesía", "entering"
        ])
        is_door_close_progress = any(word in progress for word in [
            "door close", "puerta cerca", "close door"
        ])
        crossing_text = f"{progress} {(reasoning_text or '').lower()} {(memory_text or '').lower()}"
        has_blind_crossing_hint = any(word in crossing_text for word in [
            "blind crossing",
            "blind sprint",
            "empujón final",
            "empujon final",
            "efecto túnel",
            "efecto tunel",
            "final push",
            "excepción del 30%",
            "excepcion del 30%"
        ])

        # Memoria anti-spin: permite mantener centering corto cuando estamos cerca.
        if not hasattr(self, 'anti_spin_memory'):
            self.anti_spin_memory = 0

        if distance in ["CERCA", "MUY_CERCA"]:
            self.anti_spin_memory = 8
        else:
            self.anti_spin_memory = max(0, self.anti_spin_memory - 1)

        # 0) AUTO-CROSSING: alineación propia confirmada → cruce ciego inmediato.
        # No espera al VLM: si el error lateral es pequeño y estamos MUY_CERCA, cruzamos.
        if (is_travesia_phase and
                x_dev <= 0.07 and
                distance == "MUY_CERCA" and
                not is_object_progress):
            return ControlMode.CROSSING

        # 1) CRUCE: en TRAVESÍA + centrado + avance explícito.
        # Permite activar crossing también en CERCA cuando ya está perfectamente
        # alineado, evitando bucles de "door close" sin disparar el cruce.
        crossing_ready = (
            is_travesia_phase and
            intended.startswith("A") and
            x_dev <= 0.05 and
            distance in ["CERCA", "MUY_CERCA"] and
            (is_crossing_progress or has_blind_crossing_hint or is_door_close_progress) and
            not is_object_progress
        )
        if crossing_ready:
            return ControlMode.CROSSING

        # En puerta MUY_CERCA, no avances en APPROACHING hasta que el cruce sea explícito.
        if distance == "MUY_CERCA" and not crossing_ready and not is_object_progress:
            return ControlMode.CENTERING

        # 2) Si se pierde visión (DESCONOCIDO) → SCANNING.
        # Si la distancia es conocida el VLM ve algo real, aunque esté en el extremo.
        if distance == "DESCONOCIDO":
            return ControlMode.SCANNING
        if (x < 0.20 or x > 0.80) and distance not in ["LEJOS", "CERCA", "MUY_CERCA", "MUY_LEJOS"]:
            if self.anti_spin_memory > 0 and distance in ["CERCA", "MUY_CERCA"] and not is_object_progress:
                return ControlMode.CENTERING
            return ControlMode.SCANNING

        # 3) Extremos absolutos (VLM apuntando a rotar puro) → escaneo.
        if x < 0.10 or x > 0.90:
            return ControlMode.SCANNING

        # 4) Alineación final antes de cruzar puerta.
        # Histeresis: evita flip-flop centering/approaching cerca del umbral.
        if (
            self.mode == ControlMode.CENTERING and
            distance in ["CERCA", "MUY_CERCA"] and
            x_dev > 0.04 and
            not is_object_progress
        ):
            return ControlMode.CENTERING

        if distance in ["CERCA", "MUY_CERCA"] and x_dev > 0.08 and not is_object_progress:
            return ControlMode.CENTERING

        return ControlMode.APPROACHING
        
    
    def compute_velocity(self, target_point: Tuple[float, float], 
                         distance: str, confidence: str) -> VelocityCommand:
        """
        Compute continuous velocity vector based on target point and mode.
        """
        x, y = target_point
        
        # --- DICCIONARIOS DE ESCALA (Blindados) ---
        DISTANCE_SCALES = {
            "MUY_CERCA": 0.4,
            "CERCA": 0.8,
            "LEJOS": 1.0,
            "DESCONOCIDO": 0.5
        }
        
        CONFIDENCE_SCALES = {
            "ALTA": 1.0,
            "MEDIA": 0.7,
            "BAJA": 0.4
        }
        
        x_deviation = x - 0.5
        y_factor = 1.0 - y

        params = self.MODE_PARAMS[self.mode]
        dist_scale = DISTANCE_SCALES.get(distance, 0.5)
        conf_scale = CONFIDENCE_SCALES.get(confidence, 0.5)
        ganancia_yaw = 1.0

        if self.mode == ControlMode.CROSSING:
            vx = 0.30
            vy = 0.0
            vyaw = 0.0

        elif self.mode == ControlMode.CENTERING:
            # Strafe puro cuando la puerta está descentrada; forward crece al centrar.
            # alignment_factor: 1.0 cuando centrado, 0.0 cuando x_abs >= 0.125
            vy_raw = self.KP_LATERAL * x_deviation * params["lateral_scale"]
            x_abs = abs(x_deviation)
            alignment_factor = max(0.0, 1.0 - (x_abs * 8.0))
            vx = 0.20 * alignment_factor * conf_scale
            vy = vy_raw * conf_scale * 1.5
            vyaw = x_deviation * ganancia_yaw * params["yaw_scale"]

        elif self.mode == ControlMode.SCANNING:
            vx = 0.0
            vy = 0.0
            vyaw = x_deviation * ganancia_yaw * 1.5

        else:  # APPROACHING
            vy_raw = self.KP_LATERAL * x_deviation * params["lateral_scale"]
            vy = vy_raw * dist_scale * conf_scale * 1.2
            alignment_brake = max(0.2, 1.0 - (abs(x_deviation) * 2.0))
            vx_raw = self.KP_FORWARD * y_factor * params["forward_scale"] * alignment_brake
            if abs(x_deviation) <= 0.05:
                vx_raw *= 1.5
            vx = vx_raw * dist_scale * conf_scale
            vyaw = 0.0

        # Smoothing disabled (factor=0): drone responds fully to current command
        smoothing_factor = 0.0

        vx = smoothing_factor * self.last_cmd.vx + (1 - smoothing_factor) * vx
        vy = smoothing_factor * self.last_cmd.vy + (1 - smoothing_factor) * vy
        vyaw = smoothing_factor * self.last_cmd.vyaw + (1 - smoothing_factor) * vyaw


        max_fwd = getattr(self, 'MAX_FORWARD_VEL', 0.5)
        max_lat = getattr(self, 'MAX_LATERAL_VEL', 0.3)
        max_yaw = getattr(self, 'MAX_YAW_VEL', 0.8)
        
        vx = np.clip(vx, -max_fwd * 0.2, max_fwd)
        vy = np.clip(vy, -max_lat, max_lat)
        vyaw = np.clip(vyaw, -max_yaw, max_yaw)

        self.last_cmd = VelocityCommand(vx, vy, 0.0, vyaw)
        return self.last_cmd
    
    def velocity_to_steps(self, cmd: VelocityCommand,
                          dt: float = 1.0) -> Tuple[int, list]:
        """Convert velocity command to (steps, move_vector) for move_drone."""
        move_vec = cmd.to_move_vector()
        strafe = move_vec[0]
        forward = move_vec[1]
        yaw = move_vec[2]

        # CENTERING allows stronger strafe to avoid getting stuck; other modes stay conservative

        if self.mode == ControlMode.CENTERING:
            max_strafe_component = 0.07
        else:
            max_strafe_component = 0.04
        if abs(strafe) > max_strafe_component:
            strafe = max_strafe_component if strafe > 0 else -max_strafe_component
        
        # Rebuild move vector with capped strafe
        move_vec = [strafe, forward, yaw, 0.0]
        
        # Calculate movement magnitude for steps
        # Weight lateral less to encourage more steps with smaller lateral
        total_vel = abs(forward) * 1.0 + abs(strafe) * 0.5 + abs(yaw) * 0.3
        
        # Map to steps - but limit for lateral-dominant movements
        is_lateral_dominant = abs(strafe) > abs(forward) * 0.5
        
        # === FIX: Always have minimum movement if there's any intention ===
        # Previous bug: steps=0 when velocity was too low, causing drone to freeze
        has_movement_intent = abs(strafe) > 0.001 or abs(forward) > 0.001 or abs(yaw) > 0.001
        
        if total_vel < 0.005:
            steps = 3 if has_movement_intent else 0  # Minimum 3 steps if any intent
        elif total_vel < 0.02:
            steps = 5
        elif total_vel < 0.05:
            steps = 8 if is_lateral_dominant else 10
        elif total_vel < 0.1:
            steps = 10 if is_lateral_dominant else 15
        elif total_vel < 0.2:
            steps = 12 if is_lateral_dominant else 18
        else:
            steps = 15 if is_lateral_dominant else 25
        
        # Normalize vector for direction - scale to reasonable movement magnitude.
        # IMPORTANT: In pure-centering strafes, keep micro magnitude (don't upscale to 0.1),
        # otherwise each correction becomes too aggressive and reintroduces overshoot.
        # CENTERING uses a higher scale so the diagonal advance is more noticeable
        norm_scale = 0.13 if self.mode == ControlMode.CENTERING else 0.10

        is_pure_strafe = abs(strafe) > 0.0 and abs(forward) < 0.01 and abs(yaw) < 0.01
        if is_pure_strafe:
            move_vec = [strafe, 0.0, 0.0, 0.0]
        else:
            max_val = max(abs(v) for v in move_vec) or 1.0
            if max_val > 0.01:
                move_vec = [v / max_val * norm_scale for v in move_vec]
            else:
                move_vec = [norm_scale if v > 0 else (-norm_scale if v < 0 else 0) for v in move_vec]
        
        # === SOLUCIÓN 1: Evitar Overshoot acortando el horizonte visual ===
        # Micromanage (pocos pasos) SOLO cuando está descentrado
        # === SOLUCIÓN 1: Evitar Overshoot acortando el horizonte visual ===
        # Micromanage (pocos pasos) SOLO cuando está descentrado
        if self.mode == ControlMode.CENTERING:
            # Fewer steps when strafe is maxed (large deviation) to avoid overshoot
            if abs(strafe) >= max_strafe_component:
                steps = min(steps, 5)
            else:
                steps = min(steps, 10)
        elif self.mode == ControlMode.CROSSING:
           
            steps = 8  
        elif self.mode == ControlMode.APPROACHING:
            steps = min(steps, 10)

        # Never return 0 steps — Unity freezes and VLM gets flooded with requests
        steps = max(steps, 3)
        
        return steps, move_vec
    
    def get_status(self) -> Dict:
        """Get current controller status"""
        return {
            "mode": self.mode.value,
            "last_cmd": {
                "vx": self.last_cmd.vx,
                "vy": self.last_cmd.vy,
                "vyaw": self.last_cmd.vyaw
            }
        }


class SafetyEnvelope:
    """
    Wall proximity safety layer. Analyzes edge density in the camera image
    and overrides velocity commands when collision is imminent.
    """
    
    # Physical parameters
    DRONE_WIDTH_CM = 60
    SAFE_MARGIN_CM = 35  # Minimum clearance needed
    CRITICAL_MARGIN_CM = 20  # Emergency zone
    
    # Sensor thresholds (based on edge density analysis)
    DANGER_THRESHOLD = 0.60
    CRITICAL_THRESHOLD = 0.85
    
    def __init__(self):
        from reasoning.wall_proximity_detector import WallProximityDetector
        self.wall_detector = WallProximityDetector(
            edge_threshold=40,
            imbalance_threshold=0.15  # More sensitive
        )
        self.intervention_count = 0
        self.last_analysis = None
        
    def apply_safety_envelope(self, image_cv2,
                               velocity_cmd: VelocityCommand,
                               distance: str) -> Tuple[VelocityCommand, Optional[str], Dict]:
        """Returns (safe_velocity, override_reason, analysis)."""
        analysis = self.wall_detector.analyze_proximity(image_cv2)
        self.last_analysis = analysis

        left_density = analysis["left_density"]
        right_density = analysis["right_density"]
        imbalance = analysis["imbalance"]

        if max(left_density, right_density) < 0.08:
            imbalance = 0.0

        safe_cmd = VelocityCommand(
            vx=velocity_cmd.vx,
            vy=velocity_cmd.vy,
            vz=velocity_cmd.vz,
            vyaw=velocity_cmd.vyaw
        )
        
        override_reason = None
        
        # === RULE 1: ABSOLUTE STOP if both sides critical ===
        if left_density > self.CRITICAL_THRESHOLD and right_density > self.CRITICAL_THRESHOLD:
            if velocity_cmd.vx > 0:  # Only if trying to go forward
                safe_cmd.vx = 0.0
                safe_cmd.vy = 0.0
                safe_cmd.vyaw = 0.0
                self.intervention_count += 1
                override_reason = f"EMERGENCY STOP: Both sides critical (L={left_density:.2f}, R={right_density:.2f})"
                return safe_cmd, override_reason, analysis
        
        if left_density > self.DANGER_THRESHOLD and velocity_cmd.vy < -0.05:
            correction = 0.2 * (left_density / self.CRITICAL_THRESHOLD)
            safe_cmd.vy = correction
            self.intervention_count += 1
            override_reason = f"BLOCKED LEFT: Wall detected (density={left_density:.2f}). Forcing RIGHT."

        elif right_density > self.DANGER_THRESHOLD and velocity_cmd.vy > 0.05:
            correction = -0.2 * (right_density / self.CRITICAL_THRESHOLD)
            safe_cmd.vy = correction
            self.intervention_count += 1
            override_reason = f"BLOCKED RIGHT: Wall detected (density={right_density:.2f}). Forcing LEFT."

        center_density = analysis.get("center_density", 0.0)
        if center_density > self.DANGER_THRESHOLD and velocity_cmd.vx > 0.05:
            safe_cmd.vx = 0.0
            safe_cmd.vyaw = -0.3 if left_density < right_density else 0.3
            self.intervention_count += 1
            override_reason = f"FRONTAL OBSTACLE: center_density={center_density:.2f}. Stopping forward, rotating to find path."

        elif abs(imbalance) > 0.3 and distance in ["MUY_CERCA", "CERCA"]:
            if abs(imbalance) > 0.7:
                safe_cmd.vx = velocity_cmd.vx * 0.2
                override_reason = f"PELIGRO ({imbalance:.2f}). Frenando para corregir."
            # imbalance > 0: más bordes a la izquierda → pared izquierda → forzar derecha
            if imbalance > 0:
                safe_cmd.vy = max(velocity_cmd.vy, 0.12 * abs(imbalance))
            else:
                safe_cmd.vy = min(velocity_cmd.vy, -0.12 * abs(imbalance))

        max_danger = max(left_density, right_density)
        if max_danger > self.DANGER_THRESHOLD and velocity_cmd.vx > 0.1:
            danger_factor = max(0.2, 1.0 - (max_danger / self.CRITICAL_THRESHOLD))
            safe_cmd.vx = velocity_cmd.vx * danger_factor
            if not override_reason:
                override_reason = f"SPEED REDUCED: Obstacle detected (danger={max_danger:.2f}). Factor={danger_factor:.2f}"

        if override_reason:
            self.intervention_count += 1

        return safe_cmd, override_reason, analysis

    def get_status(self) -> Dict:
        return {
            "interventions": self.intervention_count,
            "last_analysis": self.last_analysis
        }

    def reset(self):
        self.intervention_count = 0
        self.last_analysis = None
