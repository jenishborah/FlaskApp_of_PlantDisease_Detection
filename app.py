from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

model = load_model('newtry.h5')

@app.route('/')
def index():
    return render_template('index.html')


class_names = {0: 'algal', 1: 'Anthracnose', 2: 'birdeye', 3: 'brownblight', 4: 'graylight', 5: 'healthy', 6: 'redleaf', 7: 'whitespot'}

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    img = Image.open(image)
    img = img.resize((224, 224))
    img = img.convert('RGB')
    img = np.asarray(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    return predicted_class


   

if __name__ == '__main__':
    app.run(debug=True)
