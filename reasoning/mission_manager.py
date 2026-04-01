"""
Mission Manager for Drone Navigation System

Manages mission phases and transitions between them:
- BÚSQUEDA_GLOBAL: Locate door or object
- TRAVESÍA: Cross door to target room
- APROXIMACIÓN: Navigate to final object
"""

from enum import Enum


class MissionPhase(Enum):
    """Enum for different mission phases"""
    BUSQUEDA_GLOBAL = "BÚSQUEDA GLOBAL"
    TRAVESIA = "TRAVESÍA"
    APROXIMACION = "APROXIMACIÓN"
    COMPLETADA = "COMPLETADA"


class MissionManager:
    """
    Manages mission state and phase transitions.
    
    The manager tracks progress and decides when to transition between phases:
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
        self.current_phase = MissionPhase.BUSQUEDA_GLOBAL
        
        # Internal state flags
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
        
        if self.current_phase == MissionPhase.BUSQUEDA_GLOBAL:
            if not self.in_target_room:
                # Searching for door to target room
                return {
                    "phase": "BÚSQUEDA GLOBAL - Localizar Puerta",
                    "description": (
                        f"Busca visualmente una puerta abierta que conecte con '{self.target_room}'.\n"
                        f"Rota y explora la habitación hasta encontrar la puerta objetivo.\n"
                        f"Una vez detectada, comienza a acercarte progresivamente usando avances (A2/A3).\n"
                        f"NO te obsesiones con centrar desde lejos - acércate primero."
                    ),
                    "target": f"Puerta hacia {self.target_room}"
                }
            else:
                # Already in target room, searching for object
                return {
                    "phase": "BÚSQUEDA GLOBAL - Localizar Objeto",
                    "description": (
                        f"Ya estás en '{self.target_room}'. Busca visualmente '{self.target_object}'.\n"
                        f"Rota y escanea la habitación sistemáticamente hasta localizarlo.\n"
                        f"Una vez detectado, comienza a acercarte progresivamente."
                    ),
                    "target": self.target_object
                }
                
        elif self.current_phase == MissionPhase.TRAVESIA:
            # Crossing door
            return {
                "phase": "TRAVESÍA - Cruzar Puerta",
                "description": (
                    f"Cruza la puerta hacia '{self.target_room}' de forma segura.\n"
                    f"ESTRATEGIA:\n"
                    f"1. Si estás lejos (puerta <30% del frame) → AVANZA con A2 o A3\n"
                    f"2. Si estás cerca (puerta >30%) → Verifica márgenes laterales\n"
                    f"3. Si márgenes >20% en ambos lados → AVANZA con A1 o A2\n"
                    f"4. Si un lado está muy pegado → STRAFE hacia el lado con más espacio\n"
                    f"NO te obsesiones con centrar perfectamente - 20% de margen es suficiente."
                ),
                "target": "Centro del marco de la puerta"
            }
            
        elif self.current_phase == MissionPhase.APROXIMACION:
            # Approaching final object
            return {
                "phase": "APROXIMACIÓN - Alcanzar Objeto",
                "description": (
                    f"Acércate a '{self.target_object}' lo máximo posible sin colisionar.\n"
                    f"ESTRATEGIA:\n"
                    f"1. Mientras esté lejos → Avanza con A2\n"
                    f"2. Cuando esté cerca (<1m) → Usa A1 para pasos pequeños\n"
                    f"3. Si necesitas alineación lateral final → Usa D1/D2 para strafe fino\n"
                    f"Avanza progresivamente hasta estar muy cerca del objeto."
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
        
        # Update internal flags based on VLM reasoning
        if "door" in reasoning or "puerta" in reasoning:
            if "visible" in reasoning or "detected" in reasoning or "veo" in reasoning:
                if not self.door_detected:
                    self.door_detected = True
                    print(f"[MISSION] Door detected!")
                    
            if ("close" in reasoning or "cerca" in reasoning or 
                "30%" in reasoning or "40%" in reasoning):
                if not self.door_close:
                    self.door_close = True
                    print(f"[MISSION] Door is close!")
                
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
        
        if self.current_phase == MissionPhase.BUSQUEDA_GLOBAL:
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
