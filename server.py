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
    img_bytes = image_file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Call Reasoning model
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