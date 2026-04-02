import os


def load_system_prompt(path_dir, mission_phase, mission_description, target_description,
                       current_room="unknown", next_step=""):
    """
    Load unified prompt with dynamic mission injection.
    
    Args:
        path_dir: Directory containing prompt files
        mission_phase: Current mission phase (e.g., "BÚSQUEDA GLOBAL")
        mission_description: Detailed description of current mission
        target_description: Target object/location description
        current_room: Current room the drone is in (anchored context)
        next_step: What the drone should do next
        
    Returns:
        Complete system prompt with mission context injected
    """
    print(f"[PROMPT] Loading unified prompt for mission phase: {mission_phase}")
    
    try:
        # Load base prompt files
        system_base_path = os.path.join(path_dir, "drone_system_prompt.txt")
        mission_prompt_path = os.path.join(path_dir, "unified_mission_prompt.txt")
        output_prompt_path = os.path.join(path_dir, "output_prompt.txt")
        
        with open(system_base_path, "r", encoding="utf-8") as f:
            system_base = f.read()
        
        with open(mission_prompt_path, "r", encoding="utf-8") as f:
            mission_prompt = f.read()
        
        with open(output_prompt_path, "r", encoding="utf-8") as f:
            output_prompt = f.read()
        
        # Generate next_step if not provided
        if not next_step:
            next_step = _generate_next_step(mission_phase, target_description, current_room)
        
        # Inject mission context dynamically
        mission_prompt = mission_prompt.format(
            mission_phase=mission_phase,
            mission_description=mission_description,
            target_description=target_description,
            current_room=current_room,
            next_step=next_step
        )
        
        # Combine all prompts
        full_prompt = system_base + "\n\n" + mission_prompt + "\n\n" + output_prompt
        
        return full_prompt
        
    except FileNotFoundError as e:
        raise RuntimeError(f"Prompt file not found: {str(e)}")
    except KeyError as e:
        raise RuntimeError(f"Missing placeholder in mission prompt: {str(e)}")


def _generate_next_step(mission_phase, target_description, current_room):
    """
    Generate context-aware next step instruction.
    
    This helps anchor the VLM's goal and prevents it from getting confused.
    """
    phase_upper = mission_phase.upper() if mission_phase else ""
    
    # Parse target to extract object and room
    # Expected format: "object in room" (e.g., "sink in bathroom")
    target_parts = target_description.lower().split(" in ")
    target_object = target_parts[0] if target_parts else "objetivo"
    target_room = target_parts[1] if len(target_parts) > 1 else "destino"
    
    if "ESCANEO" in phase_upper:
        return f"Escanear la habitación girando para identificar puertas ABIERTAS hacia {target_room}"
    
    elif "BÚSQUEDA" in phase_upper:
        if current_room.lower() == target_room.lower():
            return f"Buscar {target_object} en esta habitación ({current_room})"
        else:
            return f"Encontrar puerta ABIERTA hacia {target_room} para llegar a {target_object}"
    
    elif "TRAVESÍA" in phase_upper:
        return f"Cruzar la puerta hacia {target_room}"
    
    elif "APROXIMACIÓN" in phase_upper:
        return f"Acercarse al {target_object}"
    
    else:
        return f"Navegar hacia {target_object} en {target_room}"