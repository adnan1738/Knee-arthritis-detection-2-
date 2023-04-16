from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

app = Flask(__name__)

dic = {0: 'Normal', 1: 'Doubtful', 2: 'Mild', 3: 'Moderate', 4: 'Severe'}

# Image Size
img_size = 256
model = load_model('model.h5')

def predict_label(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img_size, img_size))
    i = np.expand_dims(img, axis=-1) / 255.0
    i = i.reshape(1, img_size, img_size, 1)
    predict_i = model.predict(i)
    classes_i = np.argmax(predict_i, axis=1)
    return dic[classes_i[0]]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        img = request.files['file']
        img_path = "uploads/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)
        print(p)
        return str(p).lower()

if __name__ == '__main__':
    app.run(debug=True)
