
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

The project was developed in python 3.10. Please install the python dependencies:

```bash
pip install -r requirements.txt
```