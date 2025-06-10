import requests
import cv2
from io import BytesIO
import json

    
def get_cmd_vel(img, query, topology, state, addr = "http://172.18.33.110:5001"):
    url = addr + '/cmd_vel'

    # Encode image to memory
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = BytesIO(img_encoded.tobytes())

    files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
    data = {'query': query,
            'topology': json.dumps(topology),
            'state': state}

    response = requests.post(url, files=files, data=data)

    try:
        print(response.json())
    except Exception as e:
        print("Error decoding response:", e)
        print(response.text)

    return response.json()


if __name__ == "__main__":
    img = cv2.imread('test.png')
    
    query = 'Go to the kitchen'
    # Topology (example)
    topology = {
        "hallway": ["kitchen", "living room"],
        "kitchen": ["hallway"],
        "living room": ["hallway"]
    }
    state = 'Start'
    
    if img is None:
        print("Image not found.")
    else:
        get_cmd_vel(img, query, topology, state)
