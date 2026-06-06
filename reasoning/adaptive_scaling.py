"""
Adaptive Distance Scaling Module for SPF System

Adjusts movement distances dynamically based on estimated distance to target.
When far: large, fast movements (A3 = 50cm)
When near: small, precise movements (A1 = 10cm)
When very close: millimetric adjustments for safe door crossing

This reduces total flight time while ensuring smooth traversal without overshooting.
"""

from enum import Enum
from typing import Tuple, List


class DistanceLevel(Enum):
    """Distance levels estimated by VLM"""
    MUY_LEJOS = "MUY_LEJOS"      # >3m - aggressive movement
    LEJOS = "LEJOS"              # 1-3m - normal movement  
    CERCA = "CERCA"              # 0.5-1m - cautious movement
    MUY_CERCA = "MUY_CERCA"      # <0.5m - precise movement (door crossing)
    DESCONOCIDO = "DESCONOCIDO"  # Unknown - conservative default


class AdaptiveScaler:
    """
    Scales movement parameters based on distance to target.
    
    Integrates with WaypointController to provide distance-aware commands.
    """
    
    # Base movement vectors (direction only)
    BASE_VECTORS = {
        "forward": [0, 0.1, 0, 0],      # Forward (+Y)
        "rotate_left": [0, 0, -0.1, 0], # Rotate left (-yaw)
        "rotate_right": [0, 0, 0.1, 0], # Rotate right (+yaw)
        "strafe_left": [-0.1, 0, 0, 0], # Strafe left (-X)
        "strafe_right": [0.1, 0, 0, 0], # Strafe right (+X)
    }
    
    # Distance multipliers for movement magnitude
    DISTANCE_SCALES = {
        DistanceLevel.MUY_LEJOS: {
            "forward_steps": 50,      # A3 equivalent
            "rotation_degrees": 45,   # Medium rotation
            "strafe_steps": 15,       # Larger strafe
        },
        DistanceLevel.LEJOS: {
            "forward_steps": 25,      # A2 equivalent
            "rotation_degrees": 30,   # Medium-small rotation
            "strafe_steps": 10,       # Medium strafe
        },
        DistanceLevel.CERCA: {
            "forward_steps": 15,      # Between A1 and A2
            "rotation_degrees": 15,   # Small rotation
            "strafe_steps": 5,        # Small strafe
        },
        DistanceLevel.MUY_CERCA: {
            "forward_steps": 5,       # Millimetric advance
            "rotation_degrees": 5,    # Micro rotation
            "strafe_steps": 3,        # Micro strafe
        },
        DistanceLevel.DESCONOCIDO: {
            "forward_steps": 15,      # Conservative default for forward
            "rotation_degrees": 60,   # LARGER rotation for scanning (was 15)
            "strafe_steps": 5,
        },
    }
    
    # Confidence modifiers (reduce aggression when uncertain)
    CONFIDENCE_MODIFIERS = {
        "ALTA": 1.0,       # Full speed
        "MEDIA": 0.7,      # 70% speed
        "BAJA": 0.5,       # Half speed
    }
    
    def __init__(self):
        self.last_distance = DistanceLevel.DESCONOCIDO
        self.last_confidence = "MEDIA"
        
    def get_distance_level(self, distance_str: str) -> DistanceLevel:
        """Convert string distance to DistanceLevel enum"""
        try:
            return DistanceLevel(distance_str.upper())
        except (ValueError, AttributeError):
            return DistanceLevel.DESCONOCIDO
    
    def get_scaled_movement(self, 
                           movement_type: str,
                           distance: str,
                           confidence: str = "MEDIA") -> Tuple[int, List[float]]:
        """
        Get movement parameters scaled by distance and confidence.
        
        Args:
            movement_type: "forward", "rotate_left", "rotate_right", "strafe_left", "strafe_right"
            distance: Distance level string from VLM
            confidence: Confidence level from VLM
            
        Returns:
            Tuple of (steps, movement_vector) ready for move_drone()
        """
        dist_level = self.get_distance_level(distance)
        scale = self.DISTANCE_SCALES[dist_level]
        conf_mod = self.CONFIDENCE_MODIFIERS.get(confidence.upper(), 0.7)
        
        # Store for history
        self.last_distance = dist_level
        self.last_confidence = confidence
        
        # Get base vector
        base_vec = self.BASE_VECTORS.get(movement_type, [0, 0, 0, 0])
        
        # Determine steps based on movement type
        if movement_type == "forward":
            steps = int(scale["forward_steps"] * conf_mod)
        elif movement_type in ["rotate_left", "rotate_right"]:
            # For scanning (DESCONOCIDO), don't reduce rotation by confidence
            # We WANT to rotate fast to scan the room quickly
            if dist_level == DistanceLevel.DESCONOCIDO:
                steps = scale["rotation_degrees"]  # Full rotation, no confidence penalty
            else:
                steps = int(scale["rotation_degrees"] * conf_mod)
        elif movement_type in ["strafe_left", "strafe_right"]:
            steps = int(scale["strafe_steps"] * conf_mod)
        else:
            steps = 10  # Default
            
        # Ensure minimum movement
        steps = max(steps, 3)
        
        return (steps, base_vec)
    
    def command_to_movement_type(self, command: str) -> str:
        """Convert discrete command to movement type"""
        if command.startswith("A"):
            return "forward"
        elif command.startswith("B"):
            return "rotate_left"
        elif command.startswith("C"):
            return "rotate_right"
        elif command in ["D1", "D3"]:
            return "strafe_left"
        elif command in ["D2", "D4"]:
            return "strafe_right"
        else:
            return "forward"  # Default for "E" (wait/analyze)
    
    def get_adaptive_move(self, 
                          command: str, 
                          distance: str, 
                          confidence: str = "MEDIA") -> Tuple[int, List[float]]:
        """
        Convert a discrete command to adaptive movement parameters.
        
        This is the main entry point for the SPF system.
        Takes a command like "A2" and scales it based on distance.
        
        Args:
            command: Discrete command from WaypointController (e.g., "A2", "B1")
            distance: Distance level from VLM
            confidence: Confidence level from VLM
            
        Returns:
            Tuple of (steps, movement_vector) for move_drone()
        """
        if command == "E":
            # Wait/analyze - no movement
            return (0, [0, 0, 0, 0])
        
        movement_type = self.command_to_movement_type(command)
        return self.get_scaled_movement(movement_type, distance, confidence)
    
    def get_phase_description(self) -> str:
        """Get human-readable description of current scaling phase"""
        descriptions = {
            DistanceLevel.MUY_LEJOS: "Far - aggressive movement",
            DistanceLevel.LEJOS: "Medium distance - normal movement",
            DistanceLevel.CERCA: "Close - cautious movement",
            DistanceLevel.MUY_CERCA: "Very close - precise movement",
            DistanceLevel.DESCONOCIDO: "Unknown distance - conservative",
        }
        return descriptions.get(self.last_distance, "Unknown")

