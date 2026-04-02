"""
Mission Manager for Drone Navigation System

Manages mission phases and transitions between them:
- ESCANEO: Initial 360° scan of the room to identify points of interest
- BÚSQUEDA_GLOBAL: Locate door or object
- TRAVESÍA: Cross door to target room
- APROXIMACIÓN: Navigate to final object
"""

from enum import Enum


class MissionPhase(Enum):
    """Enum for different mission phases"""
    ESCANEO = "ESCANEO INICIAL"
    BUSQUEDA_GLOBAL = "BÚSQUEDA GLOBAL"
    TRAVESIA = "TRAVESÍA"
    APROXIMACION = "APROXIMACIÓN"
    COMPLETADA = "COMPLETADA"


class MissionManager:
    """
    Manages mission state and phase transitions.
    
    The manager tracks progress and decides when to transition between phases:
    0. Scan room 360° to find open doors and points of interest
    1. Search for door/object
    2. Cross door (if needed)
    3. Approach final object
    """
    
    def __init__(self, target_object, target_room):
        """
        Initialize mission manager.
        
        Args:
            target_object: Target object to reach (e.g., "sink", "red cup")
            target_room: Target room where object is located (e.g., "bathroom", "kitchen")
        """
        self.target_object = target_object
        self.target_room = target_room
        self.current_phase = MissionPhase.ESCANEO  # Start with scanning
        
        # Internal state flags
        self.scan_complete = False
        self.scan_rotations = 0  # Count rotations during scan
        self.open_door_found = False
        self.door_detected = False
        self.door_close = False
        self.crossed_door = False
        self.object_detected = False
        self.in_target_room = False
        
    def get_mission_context(self):
        """
        Generate mission context to inject into VLM prompt.
        
        Returns:
            Dict with "phase", "description", and "target" for prompt injection
        """
        
        if self.current_phase == MissionPhase.ESCANEO:
            return {
                "phase": "ESCANEO INICIAL - Explorar habitación",
                "description": (
                    f"IMPORTANTE: Antes de moverte, debes ESCANEAR la habitación.\n"
                    f"1. Gira 360° para identificar TODAS las puertas ABIERTAS.\n"
                    f"2. Las puertas CERRADAS (se ve la hoja de madera) son OBSTÁCULOS - ignóralas.\n"
                    f"3. Las puertas ABIERTAS (se ve el interior de otra habitación) son transitables.\n"
                    f"4. Busca una puerta ABIERTA que conecte con '{self.target_room}'.\n\n"
                    f"Para escanear: señala hacia los bordes (x=0.1 o x=0.9) para girar.\n"
                    f"Una vez encuentres una puerta ABIERTA, señálala para comenzar la aproximación."
                ),
                "target": f"Puerta ABIERTA hacia {self.target_room}"
            }
        
        elif self.current_phase == MissionPhase.BUSQUEDA_GLOBAL:
            if not self.in_target_room:
                # Searching for door to target room
                return {
                    "phase": "BÚSQUEDA GLOBAL - Localizar Puerta ABIERTA",
                    "description": (
                        f"Busca una puerta ABIERTA que conecte con '{self.target_room}'.\n"
                        f"RECUERDA:\n"
                        f"- Puerta ABIERTA = se ve el interior de otra habitación\n"
                        f"- Puerta CERRADA = se ve hoja de madera/metal = OBSTÁCULO, IGNORAR\n"
                        f"Una vez encontrada, acércate progresivamente señalando su centro."
                    ),
                    "target": f"Puerta ABIERTA hacia {self.target_room}"
                }
            else:
                # Already in target room, searching for object
                return {
                    "phase": "BÚSQUEDA GLOBAL - Localizar Objeto",
                    "description": (
                        f"Ya estás en '{self.target_room}'. Busca visualmente '{self.target_object}'.\n"
                        f"Gira y escanea la habitación sistemáticamente hasta localizarlo.\n"
                        f"Una vez detectado, señala su centro para comenzar la aproximación."
                    ),
                    "target": self.target_object
                }
                
        elif self.current_phase == MissionPhase.TRAVESIA:
            # Crossing door
            return {
                "phase": "TRAVESÍA - Cruzar Puerta ABIERTA",
                "description": (
                    f"Cruza la puerta ABIERTA hacia '{self.target_room}' de forma segura.\n"
                    f"ESTRATEGIA:\n"
                    f"1. Señala el CENTRO EXACTO del hueco de la puerta\n"
                    f"2. Si la puerta ocupa <40% del frame → avanza hacia ella\n"
                    f"3. Si la puerta ocupa >40% del frame → verifica que hay espacio a los lados\n"
                    f"4. Asegúrate de que hay ≥15% de margen a cada lado antes de cruzar\n"
                    f"NO intentes cruzar si ves que chocas - ajusta posición lateral primero."
                ),
                "target": "Centro del hueco de la puerta"
            }
            
        elif self.current_phase == MissionPhase.APROXIMACION:
            # Approaching final object
            return {
                "phase": "APROXIMACIÓN - Alcanzar Objeto",
                "description": (
                    f"Acércate a '{self.target_object}' señalando directamente su centro.\n"
                    f"ESTRATEGIA:\n"
                    f"1. Señala el centro del objeto para aproximarte\n"
                    f"2. Cuando estés muy cerca, señala con precisión\n"
                    f"3. No colisiones con el objeto - detente cuando estés cerca"
                ),
                "target": self.target_object
            }
            
        elif self.current_phase == MissionPhase.COMPLETADA:
            return {
                "phase": "MISIÓN COMPLETADA",
                "description": f"Has alcanzado '{self.target_object}' exitosamente. Misión cumplida.",
                "target": "Ninguno"
            }
    
    def update_phase(self, vlm_response, current_room):
        """
        Update mission phase based on VLM response and current state.
        
        Args:
            vlm_response: Dict with "reasoning", "mission_progress", etc.
            current_room: Current room name detected by the system
        """
        reasoning = vlm_response.get("reasoning", "").lower()
        progress = vlm_response.get("mission_progress", "").lower()
        target_point = vlm_response.get("target_point", [0.5, 0.5])
        confidence = vlm_response.get("confidence", "BAJA").upper()
        
        # Count scan rotations (looking at edges = scanning)
        if self.current_phase == MissionPhase.ESCANEO:
            x = target_point[0] if isinstance(target_point, (list, tuple)) else 0.5
            if x < 0.2 or x > 0.8:  # Looking at edges = scanning
                self.scan_rotations += 1
        
        # Check for open door detection
        if ("abierta" in reasoning or "open door" in reasoning or 
            "se ve el interior" in reasoning or "habitación" in reasoning):
            if not self.open_door_found:
                self.open_door_found = True
                print(f"[MISSION] Open door found!")
        
        # Update internal flags based on VLM reasoning
        if "door" in reasoning or "puerta" in reasoning:
            if "visible" in reasoning or "detected" in reasoning or "veo" in reasoning:
                if not self.door_detected:
                    self.door_detected = True
                    print(f"[MISSION] Door detected!")
                    
            if ("close" in reasoning or "cerca" in reasoning or 
                "cruzar" in reasoning or "crossing" in reasoning):
                if not self.door_close:
                    self.door_close = True
                    print(f"[MISSION] Door is close, ready to cross!")
                
        # Check if we crossed into target room
        if current_room == self.target_room and not self.in_target_room:
            self.crossed_door = True
            self.in_target_room = True
            print(f"[MISSION] Entered target room: {self.target_room}")
            
        # Check if object detected
        if self.target_object.lower() in reasoning:
            if not self.object_detected:
                self.object_detected = True
                print(f"[MISSION] Target object detected: {self.target_object}")
        
        # ==================== PHASE TRANSITIONS ====================
        
        if self.current_phase == MissionPhase.ESCANEO:
            # Transition from scan when open door found with high confidence
            if self.open_door_found and confidence in ["ALTA", "MEDIA"]:
                self.current_phase = MissionPhase.BUSQUEDA_GLOBAL
                self.scan_complete = True
                print(f"\n{'='*60}")
                print(f"[MISSION] PHASE TRANSITION: ESCANEO → BÚSQUEDA GLOBAL")
                print(f"[MISSION] Open door found, beginning approach")
                print(f"{'='*60}\n")
            # Also transition if we've scanned enough (4+ rotations)
            elif self.scan_rotations >= 6:
                self.current_phase = MissionPhase.BUSQUEDA_GLOBAL
                self.scan_complete = True
                print(f"\n{'='*60}")
                print(f"[MISSION] PHASE TRANSITION: ESCANEO → BÚSQUEDA GLOBAL")
                print(f"[MISSION] Scan complete after {self.scan_rotations} rotations")
                print(f"{'='*60}\n")
        
        elif self.current_phase == MissionPhase.BUSQUEDA_GLOBAL:
            if not self.in_target_room and self.door_detected and self.door_close:
                # Door found and close → transition to crossing
                self.current_phase = MissionPhase.TRAVESIA
                print(f"\n{'='*60}")
                print(f"[MISSION] PHASE TRANSITION: BÚSQUEDA → TRAVESÍA")
                print(f"{'='*60}\n")
                
            elif self.in_target_room and self.object_detected:
                # In target room and object found → transition to approach
                self.current_phase = MissionPhase.APROXIMACION
                print(f"\n{'='*60}")
                print(f"[MISSION] PHASE TRANSITION: BÚSQUEDA → APROXIMACIÓN")
                print(f"{'='*60}\n")
                
        elif self.current_phase == MissionPhase.TRAVESIA:
            if self.crossed_door:
                # Successfully crossed door → search for object
                self.current_phase = MissionPhase.BUSQUEDA_GLOBAL
                # Reset door flags for potential future doors
                self.door_detected = False
                self.door_close = False
                self.open_door_found = False
                print(f"\n{'='*60}")
                print(f"[MISSION] PHASE TRANSITION: TRAVESÍA → BÚSQUEDA (object)")
                print(f"{'='*60}\n")
                
        elif self.current_phase == MissionPhase.APROXIMACION:
            if "reached" in progress or "alcanzado" in progress or "arrived" in progress:
                # Object reached → mission complete
                self.current_phase = MissionPhase.COMPLETADA
                print(f"\n{'='*60}")
                print(f"[MISSION] PHASE TRANSITION: APROXIMACIÓN → COMPLETADA")
                print(f"{'='*60}\n")
    
    def is_complete(self):
        """Check if mission is complete"""
        return self.current_phase == MissionPhase.COMPLETADA
    
    def get_phase_name(self):
        """Get current phase name as string"""
        return self.current_phase.value
