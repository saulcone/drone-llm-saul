# server.py
import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
from werkzeug.utils import secure_filename
import jsonpickle
from omegaconf import OmegaConf
from reasoning.VLMModel import ReasoningModel
import json
import argparse
import logging
import os

# Importamos el detector de puertas (Safety Guard)
from reasoning.door_detector import DoorDetector

logging.basicConfig(
    format="%(message)s", 
)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("werkzeug").setLevel(logging.INFO)
logging.getLogger("openai").setLevel(logging.INFO) 
logging.getLogger("httpcore").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Load config
cfg = OmegaConf.load("config.yaml")
cfg = OmegaConf.to_container(cfg, resolve=True)
cfg = OmegaConf.create(cfg)

print("Model: ", cfg.model)
vlm_model = ReasoningModel.create_model(
    cfg.model,
    max_tokens = cfg.max_tokens,
    temperature = cfg.temperature,
    top_p = cfg.top_p,
    reasoning_effort = cfg.reasoning_effort,
)

# Inicializamos el Safety Guard (Detector de puertas con Gemini)
# No pasamos argumentos, por lo que usará Gemini por defecto según la clase que creamos antes.
print("Initializing Safety Guard (DoorDetector)...")
try:
    safety_guard = DoorDetector()
    print("Safety Guard initialized successfully.")
except Exception as e:
    safety_guard = None
    print(f"WARNING: Safety Guard (DoorDetector) failed to initialize: {e}. Safety Guard will be DISABLED.")

app = Flask(__name__)

@app.route('/cmd_vel', methods=['POST'])
def cmd_vel():
    if 'image' not in request.files or 'query' not in request.form or 'topology' not in request.form or 'state' not in request.form or 'prev_movement' not in request.form or 'mov_history' not in request.form:
        return jsonify({'error': 'Missing image, query, or topology'}), 400

    image_file = request.files['image']
    query = request.form['query']
    topology_json = request.form['topology']
    state = request.form['state']    
    previous_movement = request.form.get('prev_movement', None)
    mov_history = request.form.get('mov_history', None)
    agent_memory = request.form.get('agent_memory', '')

    try:
        topology = json.loads(topology_json)
    except Exception as e:
        return jsonify({'error': f'Invalid topology JSON: {str(e)}'}), 400

    # Convert image (Decoding)
    # IMPORTANTE: Decodificamos la imagen AQUÍ para que el Safety Guard pueda usarla
    img_bytes = image_file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # ---------------------------------------------------------
    # BLOQUE DE SEGURIDAD (SAFETY GUARD)
    # ---------------------------------------------------------
    
    # El door_detector se auto-restringe internamente por estado (solo actúa en
    # "Oriented Towards Door" y "Go Through Door"). Llamarlo siempre es seguro.
    critical_states = ["Oriented Towards Door", "Go Through Door", "Search open Door"]
    
    if state in critical_states and safety_guard is not None:
        # Preguntamos al Safety Guard (Gemini Detector) si necesitamos corregir
        safety_cmd, safety_reason = safety_guard.get_correction(img, current_state=state)
        
        if safety_cmd:
            print(f"🛡️ SAFETY INTERVENTION: {safety_reason} -> Force {safety_cmd}")
            
            # Determinamos el estado siguiente lógico tras la intervención
            next_state = state
            # Si estábamos orientados y nos manda avanzar (A1), es que ya estamos entrando
            if state == "Oriented Towards Door" and safety_cmd == "A1":
                next_state = "Go Through Door"
            
            # Construimos la respuesta manual (Override)
            override_response = {
                "room": "unknown", # No intentamos adivinar la sala durante una corrección de emergencia
                "movement": safety_cmd,
                "state": next_state,
                "description": f"[SAFETY OVERRIDE] {safety_reason}",
                "door_position": "center" if safety_cmd == "A1" else "off_center"
            }
            
            # Devolvemos la respuesta inmediatamente, SALTANDO el razonamiento normal
            safety_json = jsonpickle.encode(override_response)
            return Response(response=safety_json, status=200, mimetype="application/json")
            
    # ---------------------------------------------------------
    # FIN BLOQUE DE SEGURIDAD
    # ---------------------------------------------------------

    # Call Reasoning model (Solo se ejecuta si el Safety Guard NO intervino)
    response = vlm_model.generate_trajectory(img, query, topology=topology, 
                            state=state, previous_movement=previous_movement,
                            mov_history=mov_history,
                            agent_memory=agent_memory)
    
    gpt_response = jsonpickle.encode(response)

    return Response(response=gpt_response, status=200, mimetype="application/json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the reasoning server.')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host address to run the server on')
    parser.add_argument('--port', type=int, default=5001,
                        help='Port to run the server on')
    args = parser.parse_args()
    
    app.run(debug=True, host=args.host, port=args.port, use_reloader=False)