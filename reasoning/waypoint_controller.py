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
    # CRITICAL: Only allow advance when PERFECTLY centered (0.45-0.55)
    # This prevents collision with door frames when crossing
    X_CENTER_THRESHOLD = 0.05  # Within 0.45-0.55 is "safe to advance"
    X_STRAFE_THRESHOLD = 0.05  # Beyond ±0.05 from center needs strafe
    X_TURN_THRESHOLD = 0.40    # Beyond 0.10 or 0.90 needs rotation (edge of view)
    X_HARD_TURN_THRESHOLD = 0.45  # Beyond 0.05 or 0.95 needs hard turn
    
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
        
        # ==================== DOOR VISIBLE - PREFER STRAFE OVER ROTATION ====================
        # When door/target is visible (any distance), use STRAFE to align body
        # Only use rotation when searching (distance=DESCONOCIDO or target at extreme edge)
        
        # CRITICAL FIX: If not perfectly centered (0.45-0.55), ALWAYS strafe first
        # This prevents collision with door frames during crossing
        # X=0.65 means door is 15% to the right - MUST strafe right before advancing
        
        if abs_x_dev > self.X_CENTER_THRESHOLD and distance != "DESCONOCIDO":
            # Door visible but not centered - STRAFE to align body
            # NEVER advance until X is within ±0.05 of center
            
            if abs_x_dev > 0.25:
                # Larger misalignment - larger strafe
                if x_deviation < 0:
                    command = self.STRAFE_MAP["left_large"]  # D3: 15cm left
                else:
                    command = self.STRAFE_MAP["right_large"]  # D4: 15cm right
            elif abs_x_dev > 0.10:
                # Medium misalignment - medium strafe  
                if x_deviation < 0:
                    command = self.STRAFE_MAP["left_large"]  # D3: 15cm left
                else:
                    command = self.STRAFE_MAP["right_large"]  # D4: 15cm right
            else:
                # Small misalignment (0.05 < abs_x_dev <= 0.10) - small strafe
                if x_deviation < 0:
                    command = self.STRAFE_MAP["left_small"]  # D1: 5cm left
                else:
                    command = self.STRAFE_MAP["right_small"]  # D2: 5cm right
            
            # REMOVED: The "advance after 2 strafes" logic that caused the crash
            # The drone should NEVER advance until properly centered
            # Previous bug: After 2 strafes it would advance even with X=0.65
            
            self.last_command = command
            self.command_history.append(command)
            if len(self.command_history) > 10:
                self.command_history.pop(0)
            return command
        
        # ==================== CLOSE PROXIMITY - FINE STRAFE ====================
        # When very close, use small precise strafes for final alignment
        if distance in ["MUY_CERCA", "CERCA"] and abs_x_dev > 0.02:
            if x_deviation < 0:
                command = self.STRAFE_MAP["left_small"]  # D1: 5cm left
            else:
                command = self.STRAFE_MAP["right_small"]  # D2: 5cm right
            
            self.last_command = command
            self.command_history.append(command)
            if len(self.command_history) > 10:
                self.command_history.pop(0)
            return command
        
        # ==================== WELL CENTERED - SAFE TO ADVANCE ====================
        # ONLY advance when X is within 0.45-0.55 (±0.05 from center)
        if abs_x_dev <= self.X_CENTER_THRESHOLD:
            # Target is centered, advance based on distance
            if distance == "MUY_CERCA":
                command = "A1"
            elif distance == "CERCA":
                command = "A1"
            elif distance == "LEJOS":
                command = "A2"
            elif distance == "MUY_LEJOS":
                command = "A3"
            else:
                command = "A2"  # Default
            
            self.last_command = command
            self.command_history.append(command)
            if len(self.command_history) > 10:
                self.command_history.pop(0)
            return command
        
        # ==================== ROTATION NEEDED (searching or extreme edge) ====================
        # Only rotate when:
        # 1. Target at extreme edge (abs_x_dev >= 0.40)
        # 2. Distance is unknown (searching mode)
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
    
