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

# NOTA: Safety Guard ahora está en Simulator.py usando WallProximityDetector
# (análisis de imagen puro, sin VLM, sin latencia, sin alucinaciones de bounding boxes)

app = Flask(__name__)

@app.route('/cmd_vel', methods=['POST'])
def cmd_vel():
    # Check required fields
    required_files = ['image']
    required_form = ['query', 'topology', 'prev_movement', 'mov_history']
    
    if 'image' not in request.files:
        return jsonify({'error': 'Missing image file'}), 400
    
    for field in required_form:
        if field not in request.form:
            return jsonify({'error': f'Missing form field: {field}'}), 400

    image_file = request.files['image']
    query = request.form['query']
    topology_json = request.form['topology']
    previous_movement = request.form.get('prev_movement', None)
    mov_history = request.form.get('mov_history', None)
    agent_memory = request.form.get('agent_memory', '')
    
    # NEW: Mission parameters
    mission_phase = request.form.get('mission_phase', 'GENERAL')
    mission_description = request.form.get('mission_description', 'Navigate freely')
    target_description = request.form.get('target_description', 'Unknown')

    try:
        topology = json.loads(topology_json)
    except Exception as e:
        return jsonify({'error': f'Invalid topology JSON: {str(e)}'}), 400

    # Convert image (Decoding)
    img_bytes = image_file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Call Reasoning model with mission context
    response = vlm_model.generate_trajectory(
        img, query, 
        topology=topology,
        mission_phase=mission_phase,
        mission_description=mission_description,
        target_description=target_description,
        previous_movement=previous_movement,
        mov_history=mov_history,
        agent_memory=agent_memory
    )
    
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