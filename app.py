import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sqlite3  # <-- FIX #1: ADDED THIS MISSING IMPORT
from datetime import datetime

# Initialize the Flask App
app = Flask(__name__)

# DICTIONARY WITH DISEASE INFORMATION
disease_info = {
    "Tomato___Bacterial_spot": {
        "description": "Appears as small, water-soaked, circular spots on leaves that later turn black.",
        "treatment": "Use copper-based fungicides. Avoid overhead watering. Ensure good air circulation around plants."
    },
    "Tomato___Early_blight": {
        "description": "This fungal disease results in dark, concentric rings, often resembling a target.",
        "treatment": "Prune lower leaves. Apply fungicides containing mancozeb or chlorothalonil. Mulch the soil."
    },
    "Tomato___Late_blight": {
        "description": "A serious disease that appears as pale green, water-soaked spots, often starting at leaf edges.",
        "treatment": "Apply fungicides containing mancozeb or chlorothalonil. Remove and destroy infected plants immediately."
    },
    "Tomato___healthy": {
        "description": "The leaf appears healthy and free of common diseases.",
        "treatment": "Maintain good watering practices, ensure adequate sunlight, and monitor regularly."
    }
    # Remember to add entries for the other 6 diseases as well!
}

# NOTE: The init_db() function should be here if you want to recreate the database from scratch.
# For now, since your history.db file exists, we can omit it to keep things simple.

# Load the trained model
print("Loading the model... Please wait.")
model = load_model('tomato_model.h5')

# Load the class names
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
print("Model and class names loaded.")

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batch)
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        
        basepath = os.path.dirname(__file__)
        uploads_folder = os.path.join(basepath, 'static', 'uploads')
        os.makedirs(uploads_folder, exist_ok=True)

        file_path = os.path.join(uploads_folder, f.filename)
        f.save(file_path)

        preds = model_predict(file_path, model)
        
        CONFIDENCE_THRESHOLD = 95.0
        confidence = np.max(preds) * 100
        
        if confidence < CONFIDENCE_THRESHOLD:
            prediction_text = "Could not identify a clear tomato leaf."
            details = {
                "description": "The AI was not confident in its prediction. This can happen if the image is blurry, too dark, or not a picture of a tomato leaf.",
                "treatment": "Please try again with a clear, close-up photo of a single tomato leaf against a simple background."
            }
            print("Uncertain prediction was not saved.")
        else:
            predicted_class_index = np.argmax(preds)
            predicted_class_name = class_names[predicted_class_index]
            result = predicted_class_name.replace('___', ' ').replace('_', ' ')
            prediction_text = f"{result} ({confidence:.2f}% confidence)"
            
            details = disease_info.get(predicted_class_name, {
                "description": "No information available for this class.",
                "treatment": "Please consult a local agricultural expert."
            })

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect('history.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO scans (image_filename, prediction, confidence, scan_time) VALUES (?, ?, ?, ?)",
                           (f.filename, result, confidence, current_time))
            conn.commit()
            conn.close()
            print("A new scan was saved to the history.")
        
        return render_template('index.html', prediction=prediction_text, details=details, image_filename=f.filename)
    
    # --- FIX #2: CLEANED UP THE END OF THE FUNCTION ---
    # For a GET request (when you first load the page), just render the basic page.
    return render_template('index.html', prediction=None)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)