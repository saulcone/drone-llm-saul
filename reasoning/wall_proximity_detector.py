# filepath: c:\Users\saulc\Desktop\Cuarto\TFG\drone-llm-saul\reasoning\wall_proximity_detector.py
"""
Detector de proximidad a paredes usando análisis de imagen puro (sin VLM).
Analiza los bordes verticales de la imagen para detectar si hay paredes/marcos cerca.

SPF System Integration:
- Can be used as safety layer to prevent collisions
- Works independently of VLM decisions

Phase 4 Enhancement:
- SafetyEnvelope class provides collision prevention during door crossing
- Overrides VLM commands when physical collision is imminent
"""

import cv2
import numpy as np


class WallProximityDetector:
    """
    Detecta proximidad a paredes/marcos usando análisis de imagen.
    
    Estrategia:
    1. Divide la imagen en 3 zonas verticales: izquierda, centro, derecha
    2. Detecta bordes verticales (Sobel) que indican paredes/marcos
    3. Si hay muchos bordes en zona izquierda → pared cerca a la izquierda
    4. Si hay muchos bordes en zona derecha → pared cerca a la derecha
    5. Compara densidad de bordes para detectar desviación
    """
    
    def __init__(self, edge_threshold=50, imbalance_threshold=0.3):
        """
        Args:
            edge_threshold: Umbral para detección de bordes (Sobel)
            imbalance_threshold: Diferencia mínima entre zonas para considerar desviación (0.0-1.0)
        """
        self.edge_threshold = edge_threshold
        self.imbalance_threshold = imbalance_threshold
        
    def analyze_proximity(self, image_cv2):
        """
        Analiza la imagen para detectar proximidad a paredes.
        
        Args:
            image_cv2: Imagen BGR de OpenCV
            
        Returns:
            dict con:
                - left_density: Densidad de bordes verticales en zona izquierda (0-1)
                - right_density: Densidad de bordes verticales en zona derecha (0-1)
                - center_density: Densidad de bordes verticales en zona central (0-1)
                - imbalance: Diferencia normalizada (-1 a 1, negativo=más a izq, positivo=más a der)
                - wall_left: Boolean si hay pared cercana a la izquierda
                - wall_right: Boolean si hay pared cercana a la derecha
                - correction: Sugerencia de corrección ("left", "right", "none")
        """
        height, width = image_cv2.shape[:2]
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        
        # Aplicar blur para reducir ruido
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detectar bordes verticales usando Sobel
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        edges = np.abs(sobel_x)
        edges = (edges > self.edge_threshold).astype(np.uint8)
        
        # Dividir en 3 zonas (izquierda 30%, centro 40%, derecha 30%)
        left_zone = edges[:, :int(width * 0.30)]
        center_zone = edges[:, int(width * 0.30):int(width * 0.70)]
        right_zone = edges[:, int(width * 0.70):]
        
        # Calcular densidad de bordes en cada zona
        left_density = np.mean(left_zone)
        center_density = np.mean(center_zone)
        right_density = np.mean(right_zone)
        
        # Normalizar
        total = left_density + right_density + 0.0001  # Evitar división por cero
        left_norm = left_density / total
        right_norm = right_density / total

        if left_density < 0.03 and right_density < 0.03:
            imbalance = 0.0
        else:

            imbalance = right_norm - left_norm
        
        # Detectar paredes cercanas (alta densidad de bordes)
        high_density_threshold = 0.15  # Umbral para considerar "mucha" densidad
        wall_left = left_density > high_density_threshold
        wall_right = right_density > high_density_threshold
        
        # Sugerir corrección basada en desbalance
        correction = "none"
        if abs(imbalance) > self.imbalance_threshold:
            if imbalance < 0:
                # Más bordes a la izquierda → pared cercana a la izq → girar derecha
                correction = "right"
            else:
                # Más bordes a la derecha → pared cercana a la der → girar izquierda
                correction = "left"
        
        return {
            "left_density": float(left_density),
            "right_density": float(right_density),
            "center_density": float(center_density),
            "imbalance": float(imbalance),
            "wall_left": wall_left,
            "wall_right": wall_right,
            "correction": correction
        }
    
    def get_safe_command(self, image_cv2, intended_command):
        """
        Evalúa un comando y lo modifica si detecta riesgo de colisión.
        
        For SPF system: Only intervenes on forward commands (A*) to prevent collisions.
        
        Args:
            image_cv2: Imagen BGR de OpenCV
            intended_command: Comando original del WaypointController (ej: "A1")
            
        Returns:
            tuple: (comando_seguro, mensaje_explicativo)
        """
        # Solo evaluar comandos de avance
        if not intended_command.startswith("A"):
            return intended_command, "Non-forward command, no intervention"
        
        analysis = self.analyze_proximity(image_cv2)
        
        # Si hay desbalance significativo, corregir
        if analysis["correction"] == "left":
            return "C1", f"Wall proximity RIGHT detected (imbalance={analysis['imbalance']:.2f}). Correcting LEFT."
        elif analysis["correction"] == "right":
            return "B1", f"Wall proximity LEFT detected (imbalance={analysis['imbalance']:.2f}). Correcting RIGHT."
        
        # Si estamos centrados, permitir avance
        return intended_command, f"Path clear (imbalance={analysis['imbalance']:.2f})"


class SafetyEnvelope:
    """
    Phase 4: Virtual Safety Envelope for door crossing.
    
    Provides a physical safety layer that overrides VLM commands when
    collision risk is detected. Uses image analysis to estimate if
    drone body (including propellers) will fit through the opening.
    
    The drone is ~60cm wide, camera is centered. This means we need
    ~30cm clearance on each side to avoid collision.
    """
    
    # Drone physical parameters
    DRONE_WIDTH_CM = 60
    SAFETY_MARGIN_CM = 5  # Extra margin beyond physical width
    
    # Image analysis parameters  
    DANGER_ZONE_THRESHOLD = 0.25  # If wall is in 25% of image edge, collision likely
    CRITICAL_DENSITY_THRESHOLD = 0.20  # High edge density = close obstacle
    
    def __init__(self):
        self.wall_detector = WallProximityDetector(
            edge_threshold=40,  # More sensitive
            imbalance_threshold=0.20  # Stricter balance requirement
        )
        self.last_analysis = None
        self.intervention_count = 0
        
    def check_crossing_safety(self, image_cv2, target_point, distance, intended_command):
        """
        Check if it's safe to execute a forward command during door crossing.
        
        Args:
            image_cv2: Current camera image
            target_point: [x, y] waypoint from VLM
            distance: Distance estimate from VLM
            intended_command: Command VLM wants to execute
            
        Returns:
            tuple: (safe_command, override_reason, analysis_data)
        """
        # Only intervene on forward commands
        if not intended_command.startswith("A"):
            return intended_command, None, None
        
        # Analyze image for wall proximity - ALWAYS analyze, not just when close
        analysis = self.wall_detector.analyze_proximity(image_cv2)
        target_x = target_point[0] if target_point else 0.5
        analysis["target_x"] = target_x
        self.last_analysis = analysis
        
        # === CHECK FOR EXTREME IMBALANCE (wall on one side) ===
        # If imbalance > 0.5, there's definitely a wall close on one side
        if abs(analysis["imbalance"]) > 0.5:
            if analysis["imbalance"] < 0:
                # More edges on LEFT = wall on left = strafe RIGHT
                self.intervention_count += 1
                return "D2", f"SAFETY: Wall detected LEFT (imbalance={analysis['imbalance']:.2f}). Strafe RIGHT.", analysis
            else:
                # More edges on RIGHT = wall on right = strafe LEFT
                self.intervention_count += 1
                return "D1", f"SAFETY: Wall detected RIGHT (imbalance={analysis['imbalance']:.2f}). Strafe LEFT.", analysis
        
        # For distances other than CERCA/MUY_CERCA, only do basic imbalance check above
        if distance not in ["MUY_CERCA", "CERCA"]:
            return intended_command, None, analysis
        
        # === CLOSE PROXIMITY CHECKS ===
        
        # Check 1: Is target_point dangerously off-center? 
        if abs(target_x - 0.5) > 0.10:
            # Target is off-center - risk of collision
            if target_x < 0.5:
                self.intervention_count += 1
                return "D1", f"SAFETY: Door at X={target_x:.2f}. Strafe LEFT to align.", analysis
            else:
                self.intervention_count += 1
                return "D2", f"SAFETY: Door at X={target_x:.2f}. Strafe RIGHT to align.", analysis
        
        # Check 2: Is there high edge density in danger zones?
        left_danger = analysis["left_density"] > self.CRITICAL_DENSITY_THRESHOLD
        right_danger = analysis["right_density"] > self.CRITICAL_DENSITY_THRESHOLD
        
        if left_danger and right_danger:
            # Both sides have obstacles - tight passage
            if intended_command in ["A2", "A3"]:
                return "A1", f"SAFETY: Tight passage. Reduced to A1.", analysis
                     
        elif left_danger and not right_danger:
            self.intervention_count += 1
            return "D2", f"SAFETY: Wall LEFT (density={analysis['left_density']:.2f}). Strafe RIGHT.", analysis
            
        elif right_danger and not left_danger:
            self.intervention_count += 1
            return "D1", f"SAFETY: Wall RIGHT (density={analysis['right_density']:.2f}). Strafe LEFT.", analysis
        
        # All checks passed - safe to proceed
        return intended_command, None, analysis
    
    def get_status(self):
        """Get current safety envelope status"""
        return {
            "interventions": self.intervention_count,
            "last_analysis": self.last_analysis
        }
    
    def reset(self):
        """Reset intervention counter"""
        self.intervention_count = 0
        self.last_analysis = None


class DoorFrameAnalyzer:
    """
    Analiza el marco de la puerta para detectar si el dron está centrado.
    Usa detección de líneas verticales para encontrar los bordes del marco.
    """
    
    def __init__(self):
        self.min_line_length = 50
        self.max_line_gap = 10
        
    def find_door_frame(self, image_cv2):
        """
        Intenta detectar las líneas verticales del marco de la puerta.
        
        Returns:
            dict con:
                - left_edge_x: Posición X del borde izquierdo del marco (o None)
                - right_edge_x: Posición X del borde derecho del marco (o None)
                - frame_center_x: Centro del marco (o None si no se detecta)
                - image_center_x: Centro de la imagen
                - offset: Desviación del centro del marco respecto al centro de imagen
        """
        height, width = image_cv2.shape[:2]
        image_center_x = width // 2
        
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detectar líneas con Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=self.min_line_length,
                                maxLineGap=self.max_line_gap)
        
        if lines is None:
            return {
                "left_edge_x": None,
                "right_edge_x": None,
                "frame_center_x": None,
                "image_center_x": image_center_x,
                "offset": None
            }
        
        # Filtrar líneas verticales (ángulo cercano a 90°)
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle > 70:  # Casi vertical
                    vertical_lines.append((x1 + x2) // 2)  # Posición X promedio
        
        if len(vertical_lines) < 2:
            return {
                "left_edge_x": None,
                "right_edge_x": None,
                "frame_center_x": None,
                "image_center_x": image_center_x,
                "offset": None
            }
        
        # Encontrar las líneas verticales más a la izquierda y derecha cerca del centro
        vertical_lines.sort()
        
        # Buscar par de líneas que formen un marco plausible
        left_candidates = [x for x in vertical_lines if x < image_center_x]
        right_candidates = [x for x in vertical_lines if x > image_center_x]
        
        left_edge = max(left_candidates) if left_candidates else None
        right_edge = min(right_candidates) if right_candidates else None
        
        frame_center = None
        offset = None
        if left_edge is not None and right_edge is not None:
            frame_center = (left_edge + right_edge) // 2
            offset = frame_center - image_center_x
        
        return {
            "left_edge_x": left_edge,
            "right_edge_x": right_edge,
            "frame_center_x": frame_center,
            "image_center_x": image_center_x,
            "offset": offset
        }
