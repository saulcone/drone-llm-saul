"""
Waypoint Controller - See, Point, Fly (SPF) System

Converts VLM waypoint output (2D image coordinates) to discrete movement commands.
The VLM "points" at where it wants to go, and this controller translates that 
into appropriate drone movement commands.

Coordinate System:
- target_point[0] (x): 0.0 = left edge, 1.0 = right edge, 0.5 = center
- target_point[1] (y): 0.0 = top edge, 1.0 = bottom edge, 0.5 = center
"""

import math


class WaypointController:
    """
    Converts 2D image waypoints to drone movement commands.
    
    The controller uses simple heuristics based on:
    - X position: Determines rotation direction (left/right)
    - Y position: Determines forward/backward speed
    - Distance: Determines step size
    - Confidence: Modulates aggressiveness
    """
    
    # Thresholds for X position (horizontal)
    X_CENTER_THRESHOLD = 0.15  # Within 0.35-0.65 is "centered"
    X_TURN_THRESHOLD = 0.25    # Beyond 0.25 or 0.75 is "turn needed"
    X_HARD_TURN_THRESHOLD = 0.35  # Beyond 0.15 or 0.85 is "hard turn"
    
    # Thresholds for Y position (vertical/depth)
    Y_FORWARD_THRESHOLD = 0.55  # Below 0.55 = forward motion
    Y_BACKWARD_THRESHOLD = 0.65  # Above 0.65 = backward/stop
    
    # Distance to step size mapping
    DISTANCE_STEP_MAP = {
        "MUY_LEJOS": "A3",    # 30cm - fast approach
        "LEJOS": "A2",        # 20cm - normal approach
        "CERCA": "A1",        # 10cm - careful approach
        "MUY_CERCA": "A1",    # 10cm - very careful
        "DESCONOCIDO": "A2"   # 20cm - default
    }
    
    # Rotation commands
    ROTATION_MAP = {
        "hard_left": "B3",    # 45° left
        "left": "B2",         # 30° left
        "slight_left": "B1",  # 15° left
        "hard_right": "C3",   # 45° right
        "right": "C2",        # 30° right
        "slight_right": "C1"  # 15° right
    }
    
    # Strafe commands
    STRAFE_MAP = {
        "left_large": "D3",   # 15cm left
        "left_small": "D1",   # 5cm left
        "right_large": "D4",  # 15cm right
        "right_small": "D2"   # 5cm right
    }
    
    def __init__(self):
        """Initialize the waypoint controller."""
        self.last_command = None
        self.command_history = []
        
    def compute_command(self, target_point, distance, confidence):
        """
        Convert a waypoint to a movement command.
        
        Args:
            target_point: [x, y] normalized coordinates (0.0-1.0)
            distance: "MUY_LEJOS", "LEJOS", "CERCA", "MUY_CERCA", "DESCONOCIDO"
            confidence: "ALTA", "MEDIA", "BAJA"
            
        Returns:
            str: Movement command (e.g., "A2", "B1", "C2", "D1", "E")
        """
        x, y = target_point
        
        # Validate inputs
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        
        # Calculate deviation from center
        x_deviation = x - 0.5  # Negative = left, Positive = right
        y_deviation = y - 0.5  # Negative = up/forward, Positive = down/backward
        
        # Determine primary action based on x deviation
        abs_x_dev = abs(x_deviation)
        
        # ==================== ROTATION NEEDED ====================
        if abs_x_dev > self.X_HARD_TURN_THRESHOLD:
            # Hard turn needed (target is at extreme edge)
            if x_deviation < 0:
                command = self.ROTATION_MAP["hard_left"]
            else:
                command = self.ROTATION_MAP["hard_right"]
                
        elif abs_x_dev > self.X_TURN_THRESHOLD:
            # Normal turn needed
            if x_deviation < 0:
                command = self.ROTATION_MAP["left"]
            else:
                command = self.ROTATION_MAP["right"]
                
        elif abs_x_dev > self.X_CENTER_THRESHOLD:
            # Slight adjustment needed
            if x_deviation < 0:
                command = self.ROTATION_MAP["slight_left"]
            else:
                command = self.ROTATION_MAP["slight_right"]
                
        # ==================== CENTERED - ADVANCE OR STRAFE ====================
        else:
            # Target is roughly centered, decide based on distance and y
            
            if distance in ["CERCA", "MUY_CERCA"]:
                # Close to target - use strafe for fine alignment
                if abs_x_dev > 0.05:  # Small misalignment
                    if x_deviation < 0:
                        command = self.STRAFE_MAP["left_small"]
                    else:
                        command = self.STRAFE_MAP["right_small"]
                else:
                    # Well centered, advance carefully
                    command = "A1" if distance == "MUY_CERCA" else "A2"
                    
            else:
                # Far from target - advance based on distance
                if y > self.Y_BACKWARD_THRESHOLD:
                    # Target is below center (pointing down) - don't advance much
                    command = "E"  # Wait/analyze
                elif y > self.Y_FORWARD_THRESHOLD:
                    # Target is at center height - normal advance
                    command = self.DISTANCE_STEP_MAP.get(distance, "A2")
                else:
                    # Target is above center (pointing up) - advance
                    command = self.DISTANCE_STEP_MAP.get(distance, "A2")
        
        # ==================== CONFIDENCE MODULATION ====================
        command = self._modulate_by_confidence(command, confidence)
        
        # Update history
        self.last_command = command
        self.command_history.append(command)
        if len(self.command_history) > 10:
            self.command_history.pop(0)
            
        return command
    
    def _modulate_by_confidence(self, command, confidence):
        """
        Adjust command based on confidence level.
        
        Low confidence = more conservative movement
        High confidence = use command as-is
        """
        if confidence == "ALTA":
            return command
            
        elif confidence == "MEDIA":
            # Reduce step size for forward movements
            if command == "A3":
                return "A2"
            return command
            
        elif confidence == "BAJA":
            # Very conservative - reduce all movements
            if command == "A3":
                return "A1"
            elif command == "A2":
                return "A1"
            elif command in ["B3", "C3"]:
                # Reduce large rotations
                return command[0] + "2"  # B3->B2, C3->C2
            return command
            
        return command
    
    def check_oscillation(self):
        """
        Check if recent commands suggest oscillation.
        
        Returns:
            bool: True if oscillation detected
        """
        if len(self.command_history) < 4:
            return False
            
        recent = self.command_history[-4:]
        
        # Check for left-right oscillation
        left_count = sum(1 for c in recent if c.startswith('B') or c in ['D1', 'D3'])
        right_count = sum(1 for c in recent if c.startswith('C') or c in ['D2', 'D4'])
        
        # If roughly equal left/right movements, might be oscillating
        if left_count >= 2 and right_count >= 2:
            return True
            
        return False
    
    def get_anti_oscillation_command(self, target_point, distance):
        """
        Generate a command that breaks oscillation pattern.
        
        When oscillation is detected, prioritize forward movement
        instead of lateral adjustments.
        """
        # If we can advance, do so
        if distance in ["MUY_LEJOS", "LEJOS"]:
            return "A2"  # Just advance
        elif distance == "CERCA":
            return "A1"  # Careful advance
        else:
            return "E"  # Wait and let VLM reassess


def create_waypoint_controller():
    """Factory function to create a WaypointController instance."""
    return WaypointController()


# ==================== TESTING ====================
if __name__ == "__main__":
    controller = WaypointController()
    
    # Test cases
    test_cases = [
        # (target_point, distance, confidence, expected_behavior)
        ([0.50, 0.45], "LEJOS", "ALTA", "Advance straight"),
        ([0.20, 0.50], "LEJOS", "ALTA", "Turn left"),
        ([0.80, 0.50], "LEJOS", "ALTA", "Turn right"),
        ([0.10, 0.50], "LEJOS", "ALTA", "Hard turn left"),
        ([0.90, 0.50], "LEJOS", "ALTA", "Hard turn right"),
        ([0.45, 0.40], "CERCA", "ALTA", "Strafe left small"),
        ([0.55, 0.40], "CERCA", "ALTA", "Strafe right small"),
        ([0.50, 0.40], "MUY_CERCA", "ALTA", "Advance careful"),
        ([0.50, 0.70], "LEJOS", "ALTA", "Wait/analyze"),
        ([0.50, 0.45], "LEJOS", "BAJA", "Conservative advance"),
    ]
    
    print("Waypoint Controller Test Cases:")
    print("="*60)
    
    for target, dist, conf, expected in test_cases:
        cmd = controller.compute_command(target, dist, conf)
        print(f"Point: {target}, Dist: {dist:10s}, Conf: {conf:5s} -> {cmd:3s} ({expected})")
