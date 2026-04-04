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
    KP_LATERAL = 0.2    # Lateral velocity gain - balanced
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
            "lateral_scale": 0.4,    # Moderate lateral correction
            "yaw_scale": 0.3,        # Moderate yaw correction
        },
        ControlMode.CROSSING: {
            "forward_scale": 0.3,    # Slow forward
            "lateral_scale": 0.6,    # Good lateral correction for door alignment
            "yaw_scale": 0.1,        # Minimal yaw (avoid rotation while crossing)
        },
        ControlMode.CENTERING: {
            "forward_scale": 0.0,    # Don't advance until centered
            "lateral_scale": 0.8,    # Strong lateral correction
            "yaw_scale": 0.15,       # Minimal yaw
        }
    }
    
    # === DISTANCE SCALING ===
    # IMPORTANT: Lateral movements should NOT scale much with distance
    # because we want small, incremental corrections regardless of distance
    DISTANCE_SCALES = {
        "MUY_LEJOS": 0.8,     # Slightly reduced
        "LEJOS": 0.5,         # 50% speed (was 70%)
        "CERCA": 0.3,         # 30% speed (was 40%)
        "MUY_CERCA": 0.1,     # Very slow, precise
        "DESCONOCIDO": 0.4    # Conservative default
    }
    
    # === LATERAL MOVEMENT LIMITS ===
    # Maximum lateral movement per step in meters
    MAX_LATERAL_STEP_M = 0.15  # 15cm max per step - prevents overshooting
    
    def __init__(self):
        self.mode = ControlMode.APPROACHING
        self.last_cmd = VelocityCommand(0, 0, 0, 0)
        self.smoothing_factor = 0.3  # For velocity smoothing (0-1)
        
    def set_mode(self, mode: ControlMode):
        """Set operating mode"""
        self.mode = mode
        
    def determine_mode(self, target_point: Tuple[float, float], 
                       distance: str, mission_progress: str = "") -> ControlMode:
        """
        Automatically determine the best control mode based on context.
        """
        x, y = target_point
        x_dev = abs(x - 0.5)
        
        # Scanning mode: looking for target (extreme X values)
        if distance == "DESCONOCIDO" or x < 0.15 or x > 0.85:
            return ControlMode.SCANNING
        
        # Crossing mode: very close and reasonably centered
        if distance == "MUY_CERCA" and x_dev < 0.15:
            return ControlMode.CROSSING
        
        # Centering mode: close but not centered enough
        if distance in ["MUY_CERCA", "CERCA"] and x_dev >= 0.15:
            return ControlMode.CENTERING
        
        # Default: approaching
        return ControlMode.APPROACHING
    
    def compute_velocity(self, target_point: Tuple[float, float],
                         distance: str, confidence: str = "MEDIA") -> VelocityCommand:
        """
        Compute continuous velocity from target_point.
        
        Args:
            target_point: [x, y] normalized coordinates (0-1)
            distance: "MUY_LEJOS", "LEJOS", "CERCA", "MUY_CERCA", "DESCONOCIDO"
            confidence: "ALTA", "MEDIA", "BAJA"
            
        Returns:
            VelocityCommand with continuous velocities
        """
        x, y = target_point
        
        # Clamp to valid range
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        
        # Calculate deviations from center
        x_deviation = x - 0.5  # Negative = left, Positive = right
        y_factor = 1.0 - y     # Higher Y = closer to bottom = less forward
        
        # Get mode parameters
        self.mode = self.determine_mode(target_point, distance)
        params = self.MODE_PARAMS[self.mode]
        
        # Get distance scaling
        dist_scale = self.DISTANCE_SCALES.get(distance, 0.5)
        
        # Confidence scaling
        conf_scale = {"ALTA": 1.0, "MEDIA": 0.7, "BAJA": 0.4}.get(confidence, 0.7)
        
        # === COMPUTE RAW VELOCITIES ===
        
        # Forward velocity: based on Y position and mode
        # Y < 0.5 = target above center = more forward impulse
        if self.mode == ControlMode.SCANNING:
            vx_raw = 0.0
        else:
            vx_raw = self.KP_FORWARD * y_factor * params["forward_scale"]
        
        # Lateral velocity: proportional to X deviation
        # Negative x_deviation (target left) → negative vy (strafe left)
        vy_raw = self.KP_LATERAL * x_deviation * params["lateral_scale"]
        
        # Yaw rate: also proportional to X deviation
        # But in CROSSING mode, minimize yaw to avoid disorientation
        vyaw_raw = self.KP_YAW * x_deviation * params["yaw_scale"]
        
        # === APPLY SCALING ===
        vx = vx_raw * dist_scale * conf_scale
        vy = vy_raw * dist_scale * conf_scale
        vyaw = vyaw_raw * dist_scale  # Yaw not scaled by confidence
        
        # === CLAMP TO LIMITS ===
        vx = np.clip(vx, -self.MAX_FORWARD_VEL * 0.2, self.MAX_FORWARD_VEL)
        vy = np.clip(vy, -self.MAX_LATERAL_VEL, self.MAX_LATERAL_VEL)
        vyaw = np.clip(vyaw, -self.MAX_YAW_RATE, self.MAX_YAW_RATE)
        
        # === SMOOTH VELOCITY CHANGES ===
        # Prevents jerky motion by blending with previous command
        vx = self.smoothing_factor * vx + (1 - self.smoothing_factor) * self.last_cmd.vx
        vy = self.smoothing_factor * vy + (1 - self.smoothing_factor) * self.last_cmd.vy
        vyaw = self.smoothing_factor * vyaw + (1 - self.smoothing_factor) * self.last_cmd.vyaw
        
        cmd = VelocityCommand(vx=vx, vy=vy, vz=0.0, vyaw=vyaw)
        self.last_cmd = cmd
        
        return cmd
    
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
        # Even if not trying to move toward wall, correct if imbalanced
        elif abs(imbalance) > 0.3 and distance in ["MUY_CERCA", "CERCA"]:
            if imbalance < 0:
                # More edges on left = closer to left wall = drift right
                correction = 0.15 * abs(imbalance)
                safe_cmd.vy = max(safe_cmd.vy, correction)
                if correction > 0.1:
                    override_reason = f"DRIFT CORRECTION: Wall LEFT (imbalance={imbalance:.2f}). Adding RIGHT drift."
            else:
                # More edges on right = closer to right wall = drift left
                correction = -0.15 * abs(imbalance)
                safe_cmd.vy = min(safe_cmd.vy, correction)
                if abs(correction) > 0.1:
                    override_reason = f"DRIFT CORRECTION: Wall RIGHT (imbalance={imbalance:.2f}). Adding LEFT drift."
        
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
