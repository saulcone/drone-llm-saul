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

# ---------------------------------------------------------
# 1. PARSEAMOS LOS ARGUMENTOS PRIMERO
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='Run the reasoning server.')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address to run the server on')
parser.add_argument('--port', type=int, default=5001, help='Port to run the server on')
parser.add_argument('--model', type=str, default=None, help='Model to use (e.g., qwen3-vl:8b)')

# Usamos parse_known_args para evitar conflictos internos de Flask
args, unknown = parser.parse_known_args()

# ---------------------------------------------------------
# 2. CARGAMOS LA CONFIGURACIÓN Y EL MODELO
# ---------------------------------------------------------
cfg = OmegaConf.load("config.yaml")
cfg = OmegaConf.to_container(cfg, resolve=True)
cfg = OmegaConf.create(cfg)

# Priorizamos el argumento de la consola sobre el config.yaml
final_model = args.model if args.model else cfg.model
print(f"Model: {final_model}")

vlm_model = ReasoningModel.create_model(
    final_model,
    max_tokens = cfg.get("max_tokens", 300),
    temperature = cfg.get("temperature", 0.0),
    top_p = cfg.get("top_p", 0.2),
    reasoning_effort = cfg.get("reasoning_effort", "medium"),
)

print("Initializing Safety Guard (DoorDetector)...")
safety_guard = DoorDetector()

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

    try:
        topology = json.loads(topology_json)
    except Exception as e:
        return jsonify({'error': f'Invalid topology JSON: {str(e)}'}), 400

    # Convert image (Decoding)
    img_bytes = image_file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # ---------------------------------------------------------
    # BLOQUE DE SEGURIDAD (SAFETY GUARD)
    # ---------------------------------------------------------
    critical_states = ["Oriented Towards Door", "Go Through Door", "Search open Door"]
    
    if state in critical_states:
        safety_cmd, safety_reason = safety_guard.get_correction(img, current_state=state)
        
        if safety_cmd:
            print(f"🛡️ SAFETY INTERVENTION: {safety_reason} -> Force {safety_cmd}")
            
            next_state = state
            if state == "Oriented Towards Door" and safety_cmd == "A1":
                next_state = "Go Through Door"
            
            override_response = {
                "room": "unknown", 
                "movement": safety_cmd,
                "state": next_state,
                "description": f"[SAFETY OVERRIDE] {safety_reason}",
                "door_position": "center" if safety_cmd == "A1" else "off_center"
            }
            
            safety_json = jsonpickle.encode(override_response)
            return Response(response=safety_json, status=200, mimetype="application/json")
            
    # ---------------------------------------------------------
    # FIN BLOQUE DE SEGURIDAD
    # ---------------------------------------------------------

    # Call Reasoning model 
    response = vlm_model.generate_trajectory(img, query, topology=topology, 
                            state=state, previous_movement=previous_movement,
                            mov_history=mov_history)
    
    gpt_response = jsonpickle.encode(response)

    return Response(response=gpt_response, status=200, mimetype="application/json")


if __name__ == '__main__':
    # Los argumentos ya se han parseado arriba, simplemente usamos args.host y args.port
    app.run(debug=True, host=args.host, port=args.port, use_reloader=False)