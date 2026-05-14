
# Robot Navigation Agent with FSM and Visual Reasoning

This project implements a **visual navigation system for a drone-like robot** capable of interpreting a user's goal and autonomously navigating between rooms or finding specific objects using egocentric visual input and a topological map.

The logic is driven by a **Finite State Machine (FSM)** defined for the robot's perception and action cycle, and executed via low-level discrete movement primitives.

For simulate the environment we use Unity.


## Features

- Egocentric image processing and reasoning.
- Topological navigation based on JSON map of room connectivity.
- FSM-controlled behavior:
  - Room recognition
  - Door/object search
  - Door alignment and traversal
  - Object localization and description
- Minimal discrete motion control (move/rotate/lateral).
- Prompt-ready for integration with LLM-based planning (e.g., GPT/LLaVA/CogVLM).


## Installation

This code requires especific version of python (>=3.7.2, <3.10.0 ), because of mlagents-envs (0.29.0) is only working with those specific python version.

Please install the python dependencies:

```bash
pip install -r requirements.txt
```

## Simulated environment

Inside of build directory

⚙️ Available Arguments

Argument	Alias	Type	Description
--fast	-f	flag	Enables the drone's fast-movement mode.
--save_logs	-l	flag	Saves logs of observations during the flight.
--run_experiment	-e	flag	Runs multiple simulator repetitions in experiment mode.
--experiment_name	-n	string	Name of the experiment used to record results (default: default_experiment).
--manual	-m	flag	Enables manual control mode.
--spawn_id	-i	int	Drone spawn point (1–10). Default is -1 for random.
--num_repeats	-r	int	Number of repetitions for each spawn point (default: 1).
--addr	-a	string	Address of the server controlling the simulation (default: http://localhost:5001).  -->
--query	-q	int	Drone query ID (0–9). Default: -1.
--save_video	-v	flag	Saves a video of the simulation.
--max_steps	-s	int	Maximum number of steps to avoid infinite loops (default: 100).
--blind_cross_steps	-	int	Frames to keep blind crossing lock once triggered (default: 8).

⚠️ Potential Sources of Errors and Clarifications

- "A camera taggeg 'MainCamera' is missing" : Ensure that all cameras in the scene, the MainCamera, and the rear and front cameras of the drone
  are referenced in the CamManager2 script within the CamManager Hierarchy Element.

- If Unity is unable to generate any of the files when starting the project, you can find the complete project at this link:
  https://drive.google.com/file/d/1_Xyb0pSYdjYu-3azR9fbGEUyhfeFfP4o/view?usp=sharing

