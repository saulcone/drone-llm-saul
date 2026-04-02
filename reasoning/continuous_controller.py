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
    vy: float      # Lateral velocity (m/s), positive = right
    vz: float      # Vertical velocity (m/s), positive = up
    vyaw: float    # Yaw rate (rad/s), positive = clockwise (right)
    
    def to_move_vector(self) -> list:
        """Convert to Unity move vector format [vx, vz, vy, vyaw]"""
        return [self.vx, self.vz, self.vy, self.vyaw]
    
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
    """
    
    # === VELOCITY LIMITS (m/s and rad/s) ===
    MAX_FORWARD_VEL = 0.5      # Max forward speed
    MAX_LATERAL_VEL = 0.3      # Max strafe speed
    MAX_YAW_RATE = 0.8         # Max rotation rate (rad/s, ~45°/s)
    
    # === PROPORTIONAL GAINS ===
    # How strongly the drone reacts to deviations
    KP_LATERAL = 0.6    # Lateral velocity gain (vy per unit X deviation)
    KP_YAW = 1.2        # Yaw rate gain (vyaw per unit X deviation)
    KP_FORWARD = 0.8    # Forward velocity gain
    
    # === MODE-SPECIFIC PARAMETERS ===
    MODE_PARAMS = {
        ControlMode.SCANNING: {
            "forward_scale": 0.0,    # Don't advance while scanning
            "lateral_scale": 0.0,    # Don't strafe while scanning
            "yaw_scale": 1.5,        # Fast rotation for scanning
        },
        ControlMode.APPROACHING: {
            "forward_scale": 1.0,    # Normal forward
            "lateral_scale": 0.8,    # Good lateral correction
            "yaw_scale": 0.5,        # Moderate yaw correction
        },
        ControlMode.CROSSING: {
            "forward_scale": 0.3,    # Slow forward
            "lateral_scale": 1.2,    # Strong lateral correction (critical!)
            "yaw_scale": 0.2,        # Minimal yaw (avoid rotation while crossing)
        },
        ControlMode.CENTERING: {
            "forward_scale": 0.0,    # Don't advance until centered
            "lateral_scale": 1.5,    # Maximum lateral correction
            "yaw_scale": 0.3,        # Minimal yaw
        }
    }
    
    # === DISTANCE SCALING ===
    DISTANCE_SCALES = {
        "MUY_LEJOS": 1.0,     # Full speed
        "LEJOS": 0.7,         # 70% speed
        "CERCA": 0.4,         # 40% speed
        "MUY_CERCA": 0.15,    # Very slow, precise
        "DESCONOCIDO": 0.5    # Conservative default
    }
    
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
        
        Args:
            cmd: Velocity command
            dt: Time step (affects how far we move)
            
        Returns:
            (steps, move_vector) for move_drone function
        """
        # Calculate movement magnitude
        total_vel = cmd.magnitude() + abs(cmd.vyaw) * 0.3
        
        # Map to steps (more velocity = more steps)
        if total_vel < 0.05:
            steps = 0
        elif total_vel < 0.15:
            steps = 5
        elif total_vel < 0.3:
            steps = 10
        elif total_vel < 0.5:
            steps = 20
        else:
            steps = 30
        
        # Create move vector [vx, vz, vy, vyaw]
        # Normalize to unit vector and let steps control magnitude
        move_vec = cmd.to_move_vector()
        
        # Normalize (but keep sign)
        max_val = max(abs(v) for v in move_vec) or 1.0
        move_vec = [v / max_val for v in move_vec]
        
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
