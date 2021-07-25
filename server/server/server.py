from flask import Flask, request
import numpy as np
from model import AlexNet
import torch
import io
import logging
from PIL import Image
from util import Preprocessor, getLabel

# Start web server
app = Flask(__name__)

model = AlexNet()
model.eval()

weights_path="./pesos/alexnet-owt-4df8aa71.pth"
checkpoint = torch.load(weights_path)
model.load_state_dict(checkpoint)

if torch.cuda.is_available():
    model.to('cuda')

# Predict
@app.route('/api/predict', methods=['POST', 'GET'])
def predict():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.info('AlexNet')

    # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')

    if not file:
        return

    image_bytes = file.read()

    proc = Preprocessor()
       
    request_image = Image.open(io.BytesIO(image_bytes))
    request_image = proc.executa(request_image)
    request_image = request_image.unsqueeze(0)

    with torch.no_grad():
        saida = model(request_image)

    # Best index
    index = np.argmax(saida[0]).item() 
    accuracy = torch.max(saida).item()
    
    print(getLabel(index), accuracy)

    data = {'label':getLabel(index),'accuracy':accuracy}
    logging.info(data)

    return data
    
@app.route('/', methods=['GET'])
def index():
    return 'Image Classificator eeady to go!'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8081)