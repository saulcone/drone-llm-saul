import os


def load_system_prompt(path_dir, mission_phase, mission_description, target_description):
    """
    Load unified prompt with dynamic mission injection.
    
    Args:
        path_dir: Directory containing prompt files
        mission_phase: Current mission phase (e.g., "BÚSQUEDA GLOBAL")
        mission_description: Detailed description of current mission
        target_description: Target object/location description
        
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
        
        # Inject mission context dynamically
        mission_prompt = mission_prompt.format(
            mission_phase=mission_phase,
            mission_description=mission_description,
            target_description=target_description
        )
        
        # Combine all prompts
        full_prompt = system_base + "\n\n" + mission_prompt + "\n\n" + output_prompt
        
        return full_prompt
        
    except FileNotFoundError as e:
        raise RuntimeError(f"Prompt file not found: {str(e)}")
    except KeyError as e:
        raise RuntimeError(f"Missing placeholder in mission prompt: {str(e)}")