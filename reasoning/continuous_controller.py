# filepath: c:\Users\saulc\Desktop\Cuarto\TFG\drone-llm-saul\reasoning\continuous_controller.py
"""
Continuous Proportional Controller - Phase 4 Enhanced

Replaces discrete commands (A2, D3, C2) with continuous velocity mapping.
Takes target_point from VLM and computes proportional velocities for smooth motion.

Key principle: 
- target_point[0] (X) deviation from 0.5 → lateral velocity (vy) + yaw rate
- target_point[1] (Y) position → forward velocity (vx)
- Distance estimate → velocity scaling

This creates smooth, continuous motion that naturally centers the drone
without oscillating between discrete jumps.
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
    Proportional controller that maps target_point to continuous velocities.
    
    Instead of discrete commands like "A2" or "D3", this computes smooth
    velocities proportional to how far the target is from center.
    
    Physics model:
    - X deviation (target_x - 0.5) maps to lateral velocity AND yaw
    - Y position maps to forward velocity
    - Distance estimate scales all velocities
    
    CRITICAL: Lateral movements must be MUCH smaller than forward movements
    to prevent "slingshot" overshooting when centering doors.
    """
    
    # === VELOCITY LIMITS (m/s and rad/s) ===
    MAX_FORWARD_VEL = 0.3      # Max forward speed (reduced)
    MAX_LATERAL_VEL = 0.1      # Max strafe speed - small but not zero
    MAX_YAW_RATE = 0.5         # Max rotation rate (rad/s, ~30°/s)
    
    # === PROPORTIONAL GAINS ===
    # Balance: Not so high that we overshoot, not so low that we freeze
    KP_LATERAL = 0.3    # Lateral velocity gain - balanced
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
            "forward_scale": 0.0,    # No avanza hasta estar centrado
            "lateral_scale": 1.0,    # Fuerte corrección lateral (strafe puro)
            "yaw_scale": 0.0,        # ⛔ YAW KILL-SWITCH: Cero rotación para no torcerse
        }
    }
    
    # === DISTANCE SCALING ===
    # IMPORTANT: Lateral movements should NOT scale much with distance
    # because we want small, incremental corrections regardless of distance
    # === DISTANCE SCALING ===
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
        self.smoothing_factor = 0.3  # For velocity smoothing (0-1)
        
        # --- Estado de Memoria Anti-Amnesia ---
        self.close_memory = 0
        self.was_recently_close = False  
        self.lost_frames = 0            
        
    def set_mode(self, mode: ControlMode):
        """Set operating mode"""
        self.mode = mode
        
    def determine_mode(self, target_point: Tuple[float, float], 
                       distance: str, mission_progress: str = "") -> ControlMode:
        """Automatically determine the best control mode based on context."""
        x, y = target_point
        x_dev = abs(x - 0.5)
        
        # --- MEMORIA DIVIDIDA ---
        if not hasattr(self, 'blind_cross_memory'):
            self.blind_cross_memory = 0  
            self.anti_spin_memory = 0    
            
        # CRÍTICO: Compensar la "doble-llamada" de Simulator.py
        if distance == "MUY_CERCA" and x_dev <= 0.15:
            self.blind_cross_memory = 10  # 10 llamadas = 5 frames reales de flotación
            self.anti_spin_memory = 10
        elif distance in ["CERCA", "MUY_CERCA"]:
            self.anti_spin_memory = 10
        else:
            self.blind_cross_memory = max(0, self.blind_cross_memory - 1)
            self.anti_spin_memory = max(0, self.anti_spin_memory - 1)
            
        # 1. PÉRDIDA DE OBJETIVO o BÚSQUEDA EN EXTREMOS
        if distance == "DESCONOCIDO" or x < 0.20 or x > 0.80:
            if getattr(self, 'blind_cross_memory', 0) > 0:
                return ControlMode.CROSSING 
            if getattr(self, 'anti_spin_memory', 0) > 0:
                return ControlMode.CENTERING
            
            # Si no hay memoria de cruce, es obligatoriamente SCANNING
            return ControlMode.SCANNING
            
        # 2. OBJETIVO EN LOS EXTREMOS
        if x < 0.15 or x > 0.85:
            return ControlMode.SCANNING
            
        # 3. CENTRADO FORZOSO
        is_object = "object" in mission_progress.lower() or "sink" in mission_progress.lower()
        dynamic_threshold = 0.20 if distance in ["CERCA", "MUY_CERCA"] else 0.10
        if x_dev > dynamic_threshold and not is_object: 
            return ControlMode.CENTERING
            
        # 4. CRUCE
        if distance == "MUY_CERCA":
            return ControlMode.CROSSING
            
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
        
        # Desviaciones desde el centro
        x_deviation = x - 0.5  # Positivo = objetivo a la derecha
        y_factor = 1.0 - y     # Positivo = objetivo arriba (más rápido)
        
        params = self.MODE_PARAMS[self.mode]
        
        # Usamos los diccionarios locales
        dist_scale = DISTANCE_SCALES.get(distance, 0.5)
        conf_scale = CONFIDENCE_SCALES.get(confidence, 0.5)
        
        # --- GANANCIAS ---
        ganancia_yaw = 1.0  # Ganancia explícita para los giros en estático
        
        # Variables crudas inicializadas
        vx_raw = 0.0
        vy_raw = 0.0
        vyaw_raw = 0.0

        # === ESTRUCTURA ESTRICTA DE MODOS (Mutuamente excluyentes) ===
        if self.mode == ControlMode.CENTERING:
            vx_raw = 0.0      # FRENO FRONTAL ABSOLUTO: No avanzar mientras se centra
            vyaw_raw = 0.0    # ⛔ YAW KILL-SWITCH: Prohibido rotar para no torcerse
            vy_raw = self.KP_LATERAL * x_deviation * params["lateral_scale"]

        elif self.mode == ControlMode.CROSSING:
            vy_raw = 0.0      # BLOQUEO LATERAL: Prohibido esquivar los marcos
            vyaw_raw = 0.0    # BLOQUEO ROTACIONAL
            if distance == "MUY_CERCA":
                vx_raw = 0.30 # Empujón final potente balístico
            else:
                vx_raw = 0.15 # Velocidad de remolque a ciegas

        elif self.mode == ControlMode.SCANNING:
            vx_raw = 0.0      # FRENO FRONTAL ABSOLUTO
            vy_raw = 0.0      # NO strafe
            # Solo rotación pura (Pivot Turn de escaneo)
            vyaw_raw = x_deviation * ganancia_yaw * 1.5  

        elif self.mode == ControlMode.APPROACHING:
            # --- GIRO EN ESTÁTICO (Pivot Turn) EN ESPACIOS REDUCIDOS ---
            if abs(x_deviation) > 0.15:
                # El objeto está muy a un lado: Frena el avance y rota como una torreta
                vx_raw = 0.0
                vy_raw = 0.0
                vyaw_raw = x_deviation * ganancia_yaw
            else:
                # El objeto ya está casi en el centro: Avanza recto con micro-correcciones
                alignment_brake = max(0.2, 1.0 - (abs(x_deviation) * 2.0))
                vx_raw = self.KP_FORWARD * y_factor * params["forward_scale"] * alignment_brake
                vy_raw = self.KP_LATERAL * x_deviation * params["lateral_scale"]
                vyaw_raw = self.KP_YAW * x_deviation * params["yaw_scale"] * 0.2
                
                # Premio de velocidad por estar muy centrado
                if abs(x_deviation) <= 0.05:
                    vx_raw = vx_raw * 1.5


        # === ESCALADO DE VELOCIDADES ===
        if self.mode == ControlMode.CROSSING:
            # CROSSING es balístico, ignora escalas de confianza
            vx = vx_raw
            vy = vy_raw
            vyaw = vyaw_raw
        else:
            # Resto de modos se escalan por seguridad y visión
            vx = vx_raw * dist_scale * conf_scale
            vy = vy_raw * dist_scale * conf_scale
            vyaw = vyaw_raw * dist_scale
            

        # === SUAVIZADO Y RETENCIÓN (SMOOTHING) ===
        smoothing_factor = getattr(self, 'SMOOTHING', 0.8)

        if self.mode in [ControlMode.SCANNING, ControlMode.CENTERING]:
            self.last_cmd.vx = 0.0
            if self.mode == ControlMode.CENTERING:
                self.last_cmd.vy = 0.0  # Matar inercia lateral para no patinar

        vx = smoothing_factor * self.last_cmd.vx + (1 - smoothing_factor) * vx
        vy = smoothing_factor * self.last_cmd.vy + (1 - smoothing_factor) * vy
        vyaw = smoothing_factor * self.last_cmd.vyaw + (1 - smoothing_factor) * vyaw


        # === CLIPPING LÍMITES ===
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
        """
        Convert velocity command to discrete steps for Unity.
        
        CRITICAL: Lateral movements are capped to prevent overshooting.
        The drone should make many small lateral corrections, not
        one giant jump that overshoots the target.
        
        Args:
            cmd: Velocity command
            dt: Time step (affects how far we move)
            
        Returns:
            (steps, move_vector) for move_drone function
        """
        # Create move vector [strafe, forward, yaw, unused]
        move_vec = cmd.to_move_vector()
        
        # Get individual components
        strafe = move_vec[0]  # lateral
        forward = move_vec[1]  # forward
        yaw = move_vec[2]  # rotation
        
        # === CRITICAL: Cap lateral movement to prevent overshooting ===
        # Lateral movements should be small and incremental
        max_strafe_component = 0.05  # Very small strafe per step
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
        
        # Normalize vector for direction - scale to reasonable movement magnitude
        max_val = max(abs(v) for v in move_vec) or 1.0
        if max_val > 0.01:
            # Scale to 0.1 for consistent movement
            move_vec = [v / max_val * 0.1 for v in move_vec]
        else:
            # Very small values - give them minimum magnitude
            move_vec = [0.1 if v > 0 else (-0.1 if v < 0 else 0) for v in move_vec]
        
        # === SOLUCIÓN 1: Evitar Overshoot acortando el horizonte visual ===
        # Micromanage (pocos pasos) SOLO cuando está descentrado
        # === SOLUCIÓN 1: Evitar Overshoot acortando el horizonte visual ===
        # Micromanage (pocos pasos) SOLO cuando está descentrado
        if self.mode == ControlMode.CENTERING:
            steps = min(steps, 4)  
        elif self.mode == ControlMode.CROSSING:
           
            steps = 8  
        elif self.mode == ControlMode.APPROACHING:
            steps = min(steps, 10)

        # === SOLUCIÓN 2: Prevenir el Tiempo Congelado de Unity ===
        steps = max(steps, 3)
            
        # === SOLUCIÓN 2: Prevenir el Tiempo Congelado de Unity ===
        # Nunca devolver 0 pasos, de lo contrario la API del VLM se colapsará
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


class SafetyEnvelopeV2:
    """
    Phase 5: Enhanced Virtual Safety Envelope with absolute override.
    
    This is the "survival instinct" layer that runs locally and can
    override ANY command from the VLM if collision is imminent.
    
    Key improvements over V1:
    1. Operates on continuous velocities, not discrete commands
    2. Has absolute override authority
    3. Uses proportional response (stronger correction = closer to wall)
    4. Can STOP the drone completely if danger is extreme
    """
    
    # Physical parameters
    DRONE_WIDTH_CM = 60
    SAFE_MARGIN_CM = 35  # Minimum clearance needed
    CRITICAL_MARGIN_CM = 20  # Emergency zone
    
    # Sensor thresholds (based on edge density analysis)
    DANGER_THRESHOLD = 0.15    # Edge density indicating close obstacle
    CRITICAL_THRESHOLD = 0.25  # Edge density indicating very close obstacle
    
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
        """
        Apply safety envelope to velocity command.
        
        This is the ABSOLUTE AUTHORITY layer - it can completely override
        any velocity command if collision is detected.
        
        Args:
            image_cv2: Current camera image
            velocity_cmd: Proposed velocity from controller
            distance: Estimated distance to target
            
        Returns:
            (safe_velocity, override_reason, analysis)
        """
        # Analyze image for proximity
        analysis = self.wall_detector.analyze_proximity(image_cv2)
        self.last_analysis = analysis
        
        left_density = analysis["left_density"]
        right_density = analysis["right_density"]
        imbalance = analysis["imbalance"]
        
        # Create a copy of the velocity command
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
        
        # === RULE 2: Block dangerous lateral movement ===
        # If wall on left AND trying to go left → BLOCK and reverse
        if left_density > self.DANGER_THRESHOLD and velocity_cmd.vy < -0.05:
            # Wall on left, drone wants to go left → DANGER
            correction = 0.2 * (left_density / self.CRITICAL_THRESHOLD)
            safe_cmd.vy = correction  # Force RIGHT instead
            self.intervention_count += 1
            override_reason = f"BLOCKED LEFT: Wall detected (density={left_density:.2f}). Forcing RIGHT."
        
        # If wall on right AND trying to go right → BLOCK and reverse
        elif right_density > self.DANGER_THRESHOLD and velocity_cmd.vy > 0.05:
            # Wall on right, drone wants to go right → DANGER
            correction = -0.2 * (right_density / self.CRITICAL_THRESHOLD)
            safe_cmd.vy = correction  # Force LEFT instead
            self.intervention_count += 1
            override_reason = f"BLOCKED RIGHT: Wall detected (density={right_density:.2f}). Forcing LEFT."
        
        # === RULE 2.5: FRONTAL OBSTACLE DETECTION ===
        # If center density is high AND trying to go forward, STOP
        center_density = analysis.get("center_density", 0.0)
        if center_density > self.DANGER_THRESHOLD and velocity_cmd.vx > 0.05:
            # Obstacle directly in front - STOP forward movement
            safe_cmd.vx = 0.0
            # Maybe rotate slightly to scan for better path
            if left_density < right_density:
                safe_cmd.vyaw = -0.3  # Rotate LEFT (away from right obstacle)
            else:
                safe_cmd.vyaw = 0.3   # Rotate RIGHT (away from left obstacle)
            self.intervention_count += 1
            override_reason = f"FRONTAL OBSTACLE: center_density={center_density:.2f}. Stopping forward, rotating to find path."
        
        # === RULE 3: Proportional correction for imbalance ===
        elif abs(imbalance) > 0.3 and distance in ["MUY_CERCA", "CERCA"]:
            
            # NUEVO FRENO PARA ESQUINAS: Si roza la pared, ahogamos el acelerador al 20%
            # Esto da tiempo a que el dron se deslice lateralmente sin clavarse.
            if abs(imbalance) > 0.7:
                safe_cmd.vx = velocity_cmd.vx * 0.2
                override_reason = f"SEVERE IMBALANCE ({imbalance:.2f}). Throttling forward speed."
                
            if imbalance < 0:
                correction = 0.08 * abs(imbalance)
                safe_cmd.vy = max(safe_cmd.vy, correction)
            else:
                correction = -0.08 * abs(imbalance)
                safe_cmd.vy = min(safe_cmd.vy, correction)

        # === RULE 4: Reduce forward speed if any danger ===
        max_danger = max(left_density, right_density)
        if max_danger > self.DANGER_THRESHOLD and velocity_cmd.vx > 0.1:
            # Scale down forward speed proportionally to danger
            danger_factor = 1.0 - (max_danger / self.CRITICAL_THRESHOLD)
            danger_factor = max(0.2, danger_factor)  # Never go below 20%
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


def create_continuous_system():
    """Factory function to create the continuous control system"""
    controller = ContinuousController()
    safety = SafetyEnvelopeV2()
    return controller, safety
