from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from collections import Counter
import os
app = Flask(__name__)
model = load_model('C:\\Users\\pbsns\\OneDrive\\Documents\\cancer\\Capstone Project 5-20240505T132215Z-001\\Capstone Project 5\\code\\luk_cnn_model.keras')
input_shape = (224, 224, 3)
def preprocess_image(image):
    image = cv2.resize(image, (input_shape[0], input_shape[1]))
    image = image / 255.0  
    return image

classes = ['Benign','Malignant Early','Malignant Pre','Malignant Pro']
@app.route('/')
def upload_form():
    return render_template('upload.html')
@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    original_img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = preprocess_image(original_img)
    predictions = model.predict(np.expand_dims(img, axis=0))
    predictions = predictions.astype(int)
    first_column_values = predictions[:, :, 0].flatten()
    most_common_value, _ = Counter(first_column_values).most_common(1)[0]
    result_type = classes[most_common_value]
    temp_image_path = os.path.join('static', 'temp_image.jpg')
    cv2.imwrite(temp_image_path, original_img)
    temp_image_url = url_for('static', filename='temp_image.jpg')
    return render_template('result.html',  image=temp_image_url, result=result_type)

if __name__ == '__main__':
    app.run(debug=True)
