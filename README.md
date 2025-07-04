
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

⚙️ Argumentos disponibles
Argumento	Alias	Tipo	Descripción
--fast	-f	flag	Activa el modo de movimiento rápido del dron.
--save_logs	-l	flag	Guarda registros de las observaciones durante el vuelo.
--run_experiment	-e	flag	Ejecuta múltiples repeticiones del simulador en modo experimento.
--experiment_name	-n	string	Nombre del experimento para registrar los resultados (por defecto: default_experiment).
--manual	-m	flag	Activa el modo de control manual.
--spawn_id	-i	int	Punto de aparición del dron (1-10). Por defecto -1 para aleatorio.
--num_repeats	-r	int	Número de repeticiones de cada punto de aparición (por defecto: 1).
--addr	-a	string	Dirección del servidor que controla la simulación (por defecto: http://172.18.33.110:5001).
--query	-q	int	ID de consulta del dron (0-9). Por defecto: -1.
--save_video	-v	flag	Guarda vídeo de la simulación.
--max_steps	-s	int	Máximo número de pasos para evitar bucles infinitos (por defecto: 100).