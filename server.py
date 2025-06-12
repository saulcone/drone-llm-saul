# server.py
import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
from werkzeug.utils import secure_filename
import jsonpickle
from omegaconf import OmegaConf
from gpt_class import GPTDescriptor
import json

# Load config
cfg = OmegaConf.load("config.yaml")
cfg = OmegaConf.to_container(cfg, resolve=True)
cfg = OmegaConf.create(cfg)

gpt_model = GPTDescriptor(
    cfg.chatgpt.model,
    cfg.chatgpt.max_tokens,
    cfg.chatgpt.temperature,
    cfg.chatgpt.top_p
)

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

    # Convert image
    img_bytes = image_file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Call GPT
    response = gpt_model.generate_trajectory(img, query, topology=topology, 
                                             state=state, 
                                             previous_movement=previous_movement,
                                             mov_history=mov_history)
    gpt_response = jsonpickle.encode(response)

    return Response(response=gpt_response, status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)
