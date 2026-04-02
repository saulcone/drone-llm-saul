"""
Model Predictive Controller (MPC) for Drone Navigation

This controller provides smooth, continuous trajectory planning from VLM waypoints.
Instead of discrete commands (A1, B2, C3), it generates continuous velocity commands
that result in smooth curves, especially useful for door crossing.

Key Features:
- Converts image-space waypoints to velocity commands
- Trajectory smoothing with exponential decay
- Anti-oscillation through momentum-based control
- Adaptive velocity based on distance to target
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math


class ControlMode(Enum):
    """Operating modes for the MPC controller"""
    SCANNING = "scanning"       # Rotating to scan environment
    APPROACHING = "approaching" # Moving toward target
    CROSSING = "crossing"       # Precise door crossing
    HOVERING = "hovering"       # Stationary, waiting


@dataclass
class DroneState:
    """Current state of the drone"""
    x: float = 0.0           # Position X
    y: float = 0.0           # Position Y (forward)
    z: float = 0.0           # Position Z (height)
    yaw: float = 0.0         # Heading angle (degrees)
    vx: float = 0.0          # Velocity X
    vy: float = 0.0          # Velocity Y
    vyaw: float = 0.0        # Angular velocity


@dataclass
class VelocityCommand:
    """Velocity command output for the drone"""
    vx: float = 0.0          # Strafe velocity (left/right)
    vy: float = 0.0          # Forward velocity
    vyaw: float = 0.0        # Yaw rate (rotation)
    vz: float = 0.0          # Vertical velocity (altitude)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for Unity interface"""
        return np.array([[self.vx, self.vy, self.vyaw, self.vz, 0]], dtype=np.float32)
    
    def to_list(self) -> List[float]:
        """Convert to list for compatibility"""
        return [self.vx, self.vy, self.vyaw, self.vz]


class MPCController:
    """
    Model Predictive Controller for smooth drone navigation.
    
    Converts VLM waypoints (image coordinates) to continuous velocity commands,
    generating smooth trajectories instead of discrete movements.
    """
    
    # Control parameters
    MAX_FORWARD_VELOCITY = 0.15      # Max forward speed
    MAX_STRAFE_VELOCITY = 0.10       # Max lateral speed
    MAX_YAW_RATE = 0.12              # Max rotation rate (rad/s equivalent)
    
    # Image coordinate thresholds
    CENTER_THRESHOLD = 0.15          # How close to center is "centered" (0.35-0.65)
    EDGE_THRESHOLD = 0.25            # How close to edge triggers rotation
    
    # Smoothing parameters
    VELOCITY_SMOOTHING = 0.7         # Exponential smoothing factor (0-1, higher = smoother)
    MOMENTUM_DECAY = 0.85            # How quickly momentum decays
    
    # Distance-based scaling
    DISTANCE_SCALES = {
        "MUY_LEJOS": 1.0,            # Full speed
        "LEJOS": 0.7,                 # 70% speed
        "CERCA": 0.4,                 # 40% speed
        "MUY_CERCA": 0.2,             # 20% speed - precision mode
        "DESCONOCIDO": 0.5,           # Conservative
    }
    
    def __init__(self):
        # State tracking
        self.current_mode = ControlMode.SCANNING
        self.state = DroneState()
        
        # Velocity history for smoothing
        self.last_cmd = VelocityCommand()
        self.velocity_buffer: List[VelocityCommand] = []
        self.buffer_size = 5
        
        # Trajectory planning
        self.target_point: Optional[Tuple[float, float]] = None
        self.trajectory_progress = 0.0
        
        # Anti-oscillation
        self.rotation_direction = 0  # -1 left, 0 none, 1 right
        self.rotation_momentum = 0.0
        self.direction_lock_counter = 0
        
    def compute_velocity(self, 
                        target_point: Tuple[float, float],
                        distance: str = "DESCONOCIDO",
                        confidence: str = "MEDIA",
                        is_scanning: bool = False) -> VelocityCommand:
        """
        Compute smooth velocity command from image waypoint.
        
        Args:
            target_point: (x, y) normalized coordinates [0,1] in image space
                         x=0 is left edge, x=1 is right edge
                         y=0 is top, y=1 is bottom
            distance: Estimated distance to target
            confidence: Confidence level of VLM
            is_scanning: If True, in scanning mode (prioritize rotation)
            
        Returns:
            VelocityCommand with smooth velocities
        """
        x, y = target_point
        
        # Update mode based on context
        self._update_mode(x, y, distance, is_scanning)
        
        # Get distance scale factor
        dist_scale = self.DISTANCE_SCALES.get(distance.upper(), 0.5)
        
        # Compute raw velocities based on waypoint position
        raw_cmd = self._compute_raw_velocity(x, y, dist_scale, is_scanning)
        
        # Apply smoothing
        smooth_cmd = self._apply_smoothing(raw_cmd)
        
        # Apply anti-oscillation
        final_cmd = self._apply_anti_oscillation(smooth_cmd)
        
        # Update state
        self.last_cmd = final_cmd
        self.target_point = target_point
        
        return final_cmd
    
    def _update_mode(self, x: float, y: float, distance: str, is_scanning: bool):
        """Update control mode based on current situation"""
        if is_scanning or distance == "DESCONOCIDO":
            self.current_mode = ControlMode.SCANNING
        elif distance == "MUY_CERCA":
            self.current_mode = ControlMode.CROSSING
        elif abs(x - 0.5) > self.CENTER_THRESHOLD:
            # Target not centered, need to rotate/approach
            self.current_mode = ControlMode.APPROACHING
        else:
            self.current_mode = ControlMode.APPROACHING
    
    def _compute_raw_velocity(self, x: float, y: float, 
                              dist_scale: float, 
                              is_scanning: bool) -> VelocityCommand:
        """
        Compute raw velocity from waypoint position.
        
        The key insight: waypoint X position determines rotation,
        Y position determines forward speed.
        """
        cmd = VelocityCommand()
        
        # === YAW (ROTATION) ===
        # X deviation from center determines rotation direction and speed
        x_error = x - 0.5  # Negative = left, Positive = right
        
        if is_scanning or self.current_mode == ControlMode.SCANNING:
            # In scanning mode, rotate faster and ignore forward
            if abs(x_error) > 0.2:  # Target is significantly off-center
                cmd.vyaw = np.sign(x_error) * self.MAX_YAW_RATE * 1.5
            else:
                # Small rotation for fine adjustment during scan
                cmd.vyaw = x_error * self.MAX_YAW_RATE * 2.0
            cmd.vy = 0  # Don't advance while scanning
            return cmd
        
        # Normal mode: proportional rotation
        if abs(x_error) > self.CENTER_THRESHOLD:
            # Target not centered - rotate toward it
            # Proportional control with saturation
            cmd.vyaw = np.clip(
                x_error * self.MAX_YAW_RATE * 2.5,
                -self.MAX_YAW_RATE,
                self.MAX_YAW_RATE
            )
        else:
            # Centered enough - minimal rotation correction
            cmd.vyaw = x_error * self.MAX_YAW_RATE * 0.5
        
        # === FORWARD VELOCITY ===
        # Y position determines forward speed (lower Y = further away = faster)
        # Also scale by distance estimate
        
        if self.current_mode == ControlMode.CROSSING:
            # Precision mode for door crossing
            cmd.vy = self.MAX_FORWARD_VELOCITY * 0.3 * dist_scale
        elif abs(x_error) < self.CENTER_THRESHOLD * 1.5:
            # Target reasonably centered - can advance
            # Speed inversely related to y (higher y = closer = slower)
            forward_factor = 1.0 - (y * 0.5)  # y=0 -> 1.0, y=1 -> 0.5
            cmd.vy = self.MAX_FORWARD_VELOCITY * forward_factor * dist_scale
        else:
            # Target far from center - prioritize rotation, slow forward
            cmd.vy = self.MAX_FORWARD_VELOCITY * 0.2 * dist_scale
        
        # === STRAFE (LATERAL MOVEMENT) ===
        # Only used for fine adjustments during door crossing
        if self.current_mode == ControlMode.CROSSING and abs(x_error) > 0.05:
            cmd.vx = np.clip(
                x_error * self.MAX_STRAFE_VELOCITY * 2,
                -self.MAX_STRAFE_VELOCITY,
                self.MAX_STRAFE_VELOCITY
            )
        
        return cmd
    
    def _apply_smoothing(self, raw_cmd: VelocityCommand) -> VelocityCommand:
        """
        Apply exponential smoothing to velocity commands.
        
        This prevents jerky movements by gradually transitioning
        from current velocity to target velocity.
        """
        alpha = 1.0 - self.VELOCITY_SMOOTHING
        
        smooth_cmd = VelocityCommand(
            vx=alpha * raw_cmd.vx + self.VELOCITY_SMOOTHING * self.last_cmd.vx,
            vy=alpha * raw_cmd.vy + self.VELOCITY_SMOOTHING * self.last_cmd.vy,
            vyaw=alpha * raw_cmd.vyaw + self.VELOCITY_SMOOTHING * self.last_cmd.vyaw,
            vz=alpha * raw_cmd.vz + self.VELOCITY_SMOOTHING * self.last_cmd.vz,
        )
        
        # Add to buffer for history
        self.velocity_buffer.append(smooth_cmd)
        if len(self.velocity_buffer) > self.buffer_size:
            self.velocity_buffer.pop(0)
        
        return smooth_cmd
    
    def _apply_anti_oscillation(self, cmd: VelocityCommand) -> VelocityCommand:
        """
        Apply anti-oscillation logic to prevent left-right hunting.
        
        Uses momentum-based direction locking to prevent rapid direction changes.
        """
        # Track rotation direction
        if abs(cmd.vyaw) > 0.01:
            new_direction = np.sign(cmd.vyaw)
            
            # Check for direction reversal
            if new_direction != self.rotation_direction and self.rotation_direction != 0:
                # Direction is reversing
                if self.direction_lock_counter > 0:
                    # Still locked - dampen the reversal
                    cmd.vyaw *= 0.3
                    self.direction_lock_counter -= 1
                else:
                    # Allow direction change but apply momentum decay
                    self.rotation_momentum *= self.MOMENTUM_DECAY
                    self.direction_lock_counter = 3  # Lock for 3 steps
            else:
                # Same direction - build momentum
                self.rotation_momentum = min(1.0, self.rotation_momentum + 0.2)
                self.direction_lock_counter = max(0, self.direction_lock_counter - 1)
            
            self.rotation_direction = int(new_direction)
        else:
            # No significant rotation - decay momentum
            self.rotation_momentum *= 0.8
            if self.rotation_momentum < 0.1:
                self.rotation_direction = 0
        
        return cmd
    
    def get_movement_steps(self, cmd: VelocityCommand, 
                          base_steps: int = 10) -> Tuple[int, List[float]]:
        """
        Convert velocity command to discrete steps for Unity interface.
        
        This bridges the continuous MPC output with the discrete
        step-based Unity interface.
        
        Args:
            cmd: Velocity command
            base_steps: Base number of simulation steps
            
        Returns:
            Tuple of (steps, movement_vector)
        """
        # Determine step count based on velocity magnitude
        velocity_magnitude = math.sqrt(cmd.vx**2 + cmd.vy**2 + cmd.vyaw**2)
        
        if velocity_magnitude < 0.01:
            return (0, [0, 0, 0, 0])
        
        # Scale steps by velocity (more velocity = more steps for same result)
        # But cap to reasonable range
        steps = int(base_steps * (1 + velocity_magnitude * 5))
        steps = np.clip(steps, 5, 60)
        
        # Movement vector
        move_vec = cmd.to_list()
        
        return (steps, move_vec)
    
    def reset(self):
        """Reset controller state"""
        self.current_mode = ControlMode.SCANNING
        self.state = DroneState()
        self.last_cmd = VelocityCommand()
        self.velocity_buffer.clear()
        self.target_point = None
        self.rotation_direction = 0
        self.rotation_momentum = 0.0
        self.direction_lock_counter = 0
    
    def get_status(self) -> dict:
        """Get current controller status for debugging"""
        return {
            "mode": self.current_mode.value,
            "target": self.target_point,
            "last_cmd": {
                "vx": round(self.last_cmd.vx, 3),
                "vy": round(self.last_cmd.vy, 3),
                "vyaw": round(self.last_cmd.vyaw, 3),
            },
            "rotation_direction": self.rotation_direction,
            "momentum": round(self.rotation_momentum, 2),
        }


class TrajectoryPlanner:
    """
    Higher-level trajectory planning for complex maneuvers.
    
    Generates waypoint sequences for:
    - Smooth door approaches
    - S-curve door crossing
    - Obstacle avoidance paths
    """
    
    def __init__(self, mpc: MPCController):
        self.mpc = mpc
        self.waypoint_queue: List[Tuple[float, float]] = []
        self.current_waypoint_idx = 0
        
    def plan_door_crossing(self, 
                          door_center: Tuple[float, float],
                          door_width_ratio: float = 0.3) -> List[Tuple[float, float]]:
        """
        Generate waypoints for smooth door crossing.
        
        Creates an S-curve path through the door to avoid
        clipping the door frame.
        
        Args:
            door_center: (x, y) of door center in image
            door_width_ratio: Estimated door width as fraction of image
            
        Returns:
            List of waypoints forming S-curve through door
        """
        cx, cy = door_center
        
        # Pre-approach: align with door center
        waypoints = [
            (cx, 0.6),      # Approach point
            (cx, 0.5),      # Door entrance
            (cx, 0.4),      # Through door
            (0.5, 0.35),    # Exit and center
        ]
        
        return waypoints
    
    def plan_scan_rotation(self, 
                          direction: str = "right",
                          segments: int = 6) -> List[Tuple[float, float]]:
        """
        Generate waypoints for 360° room scan.
        
        Args:
            direction: "left" or "right"
            segments: Number of rotation segments
            
        Returns:
            List of waypoints at image edges for rotation
        """
        if direction == "right":
            x = 0.9
        else:
            x = 0.1
            
        # Repeated edge waypoints for rotation
        return [(x, 0.5)] * segments
    
    def get_next_waypoint(self) -> Optional[Tuple[float, float]]:
        """Get next waypoint from queue"""
        if self.current_waypoint_idx < len(self.waypoint_queue):
            wp = self.waypoint_queue[self.current_waypoint_idx]
            self.current_waypoint_idx += 1
            return wp
        return None
    
    def set_waypoints(self, waypoints: List[Tuple[float, float]]):
        """Set new waypoint queue"""
        self.waypoint_queue = waypoints
        self.current_waypoint_idx = 0
    
    def clear_waypoints(self):
        """Clear waypoint queue"""
        self.waypoint_queue.clear()
        self.current_waypoint_idx = 0


# Convenience function for integration
def create_mpc_controller() -> MPCController:
    """Factory function to create MPC controller instance"""
    return MPCController()


def compute_smooth_movement(target_point: Tuple[float, float],
                           distance: str = "DESCONOCIDO",
                           confidence: str = "MEDIA",
                           is_scanning: bool = False,
                           controller: Optional[MPCController] = None) -> Tuple[int, List[float]]:
    """
    Convenience function to compute smooth movement from waypoint.
    
    This is the main entry point for integration with Simulator.py
    
    Args:
        target_point: (x, y) normalized image coordinates
        distance: Distance estimate from VLM
        confidence: Confidence level
        is_scanning: If in scanning mode
        controller: Optional existing controller (will create new if None)
        
    Returns:
        Tuple of (steps, movement_vector) for move_drone()
    """
    if controller is None:
        controller = MPCController()
    
    cmd = controller.compute_velocity(
        target_point=target_point,
        distance=distance,
        confidence=confidence,
        is_scanning=is_scanning
    )
    
    return controller.get_movement_steps(cmd)
