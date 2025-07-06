import os


def load_system_prompt(path_dir, system_prompt_path,
                       output_prompt_path, curr_state):
    """Load the system prompt from a file."""
    print("current state: " + curr_state)
    if curr_state == "Recognize Room":
        state_prompt_path = "state_0.txt"
    elif curr_state == "Position in the center of room":
        state_prompt_path = "state_1.txt"
    elif curr_state == "Stay on Room":
        state_prompt_path = "state_2.txt"
    elif curr_state == "Search Object":
        state_prompt_path = "state_3.txt"
    elif curr_state == "Reach Object":
        state_prompt_path = "state_4.txt"
    elif curr_state == "Describe Object":
        state_prompt_path = "state_5.txt"
    elif curr_state == "Search open Door":
        state_prompt_path = "state_6.txt"
    elif curr_state == "Oriented Towards Door":
        state_prompt_path = "state_7.txt"
    elif curr_state == "Go Through Door":
        state_prompt_path = "state_8.txt"
    try:
        system_prompt_path = os.path.join(path_dir, system_prompt_path)
        state_prompt_path = os.path.join(path_dir, state_prompt_path)
        output_prompt_path = os.path.join(path_dir, output_prompt_path)
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt =  f.read()
        with open(state_prompt_path, "r", encoding="utf-8") as f:
            state_prompt =  f.read()
        with open(output_prompt_path, "r", encoding="utf-8") as f:
            output_prompt =  f.read()        
        prompt = system_prompt + "\n" + state_prompt + "\n" + output_prompt
        return prompt
    except FileNotFoundError:
        raise RuntimeError(f"System prompt file not found at {prompt}")