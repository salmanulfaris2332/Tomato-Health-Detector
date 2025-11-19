import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import sqlite3
from datetime import datetime
import google.generativeai as genai
import json 
import PIL.Image

# Initialize the Flask App
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit

# --- GEMINI AI CONFIGURATION ---
try:
    # NEW: Try to get key from Environment (Render), otherwise use the hardcoded one (for local testing)
    api_key = os.environ.get("GOOGLE_API_KEY") or "AIzaSyBsVEalTK-Zdw_3OfxJSWdTVCunO2K0few"
    
    genai.configure(api_key=api_key)
    chatbot_model = genai.GenerativeModel('gemini-flash-latest')
    print("Gemini AI model loaded successfully.")
except Exception as e:
    print(f"Error loading Gemini AI model: {e}")
    chatbot_model = None

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_filename TEXT NOT NULL,
        prediction TEXT NOT NULL,
        confidence REAL NOT NULL,
        scan_time TEXT NOT NULL
    );
    ''')
    conn.commit()
    conn.close()
    print("Database and table initialized.")

# --- DISEASE INFORMATION ---
all_disease_info = {
    "tomato": {
        "Tomato___Bacterial_spot": { "display_name": "Bacterial Spot", "scientific_name": "Xanthomonas campestris", "severity": "MODERATE", "spread": "Rain, Tools", "conditions": "High humidity, Warm.", "cause": "Bacteria", "symptoms": "Small, dark, water-soaked spots.", "prevention": "Certified seeds, Rotate crops.", "treatment_organic": "Copper bactericides.", "treatment_chemical": "Mancozeb + Copper." },
        "Tomato___Early_blight": { "display_name": "Early Blight", "scientific_name": "Alternaria solani", "severity": "MODERATE", "spread": "Wind, Rain", "conditions": "Wet/Dry cycles, Warm.", "cause": "Fungus", "symptoms": "Bull's-eye target spots.", "prevention": "Mulch soil, Prune lower leaves.", "treatment_organic": "Neem oil, Bio-fungicide.", "treatment_chemical": "Chlorothalonil, Mancozeb." },
        "Tomato___Late_blight": { "display_name": "Late Blight", "scientific_name": "Phytophthora infestans", "severity": "HIGH (CRITICAL)", "spread": "Wind, Water", "conditions": "Cool & Wet.", "cause": "Water Mold", "symptoms": "Greasy spots, White fuzz.", "prevention": "Air flow, Water AM.", "treatment_organic": "Copper (preventative).", "treatment_chemical": "Propamocarb, Cymoxanil." },
        "Tomato___Leaf_Mold": { "display_name": "Leaf Mold", "scientific_name": "Passalora fulva", "severity": "LOW", "spread": "Air, Tools", "conditions": "High Humidity.", "cause": "Fungus", "symptoms": "Yellow spots top, olive mold bottom.", "prevention": "Ventilation.", "treatment_organic": "Sulfur sprays.", "treatment_chemical": "Difenoconazole." },
        "Tomato___Septoria_leaf_spot": { "display_name": "Septoria Leaf Spot", "scientific_name": "Septoria lycopersici", "severity": "MODERATE", "spread": "Water, Insects", "conditions": "Wet leaves.", "cause": "Fungus", "symptoms": "Tiny spots with gray centers.", "prevention": "Remove debris, Mulch.", "treatment_organic": "Copper.", "treatment_chemical": "Chlorothalonil." },
        "Tomato___Spider_mites Two-spotted_spider_mite": { "display_name": "Spider Mites", "scientific_name": "Tetranychus urticae", "severity": "HIGH (Fast)", "spread": "Wind, Crawling", "conditions": "Hot & Dry.", "cause": "Pest", "symptoms": "Stippling dots, webbing.", "prevention": "Mist leaves.", "treatment_organic": "Neem oil, Water blast.", "treatment_chemical": "Abamectin." },
        "Tomato___Target_Spot": { "display_name": "Target Spot", "scientific_name": "Corynespora cassiicola", "severity": "MODERATE", "spread": "Airborne", "conditions": "Humid, Warm.", "cause": "Fungus", "symptoms": "Target rings with halo.", "prevention": "Air flow.", "treatment_organic": "Copper.", "treatment_chemical": "Azoxystrobin." },
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": { "display_name": "Yellow Leaf Curl", "scientific_name": "TYLCV", "severity": "HIGH (CRITICAL)", "spread": "Whiteflies", "conditions": "Warm.", "cause": "Virus", "symptoms": "Yellow curled leaves, stunted.", "prevention": "Nettings, Resistant varieties.", "treatment_organic": "Control whiteflies.", "treatment_chemical": "Imidacloprid (for flies)." },
        "Tomato___Tomato_mosaic_virus": { "display_name": "Mosaic Virus", "scientific_name": "ToMV", "severity": "HIGH (No Cure)", "spread": "Hands, Tools", "conditions": "Any.", "cause": "Virus", "symptoms": "Mottled green pattern.", "prevention": "Disinfect tools.", "treatment_organic": "Remove plant.", "treatment_chemical": "None." },
        "Tomato___healthy": { "display_name": "Healthy", "scientific_name": "Solanum lycopersicum", "severity": "NONE", "spread": "N/A", "conditions": "Optimal.", "cause": "N/A", "symptoms": "Green and clean.", "prevention": "Maintain care.", "treatment_organic": "N/A", "treatment_chemical": "N/A" }
    },
    "potato": {
        "Potato___Early_blight": { "display_name": "Early Blight", "scientific_name": "Alternaria solani", "severity": "MODERATE", "spread": "Wind, Rain", "conditions": "Aging plants.", "cause": "Fungus", "symptoms": "Dark rings on lower leaves.", "prevention": "Rotation.", "treatment_organic": "Copper.", "treatment_chemical": "Mancozeb." },
        "Potato___Late_blight": { "display_name": "Late Blight", "scientific_name": "Phytophthora infestans", "severity": "HIGH (CRITICAL)", "spread": "Wind, Tubers", "conditions": "Cool & Wet.", "cause": "Water Mold", "symptoms": "Black spots, rotting.", "prevention": "Certified seed.", "treatment_organic": "Copper.", "treatment_chemical": "Metalaxyl." },
        "Potato___healthy": { "display_name": "Healthy", "scientific_name": "Solanum tuberosum", "severity": "NONE", "spread": "N/A", "conditions": "Optimal.", "cause": "N/A", "symptoms": "Vigorous growth.", "prevention": "Watering.", "treatment_organic": "N/A", "treatment_chemical": "N/A" }
    }
}

# --- LOAD MODELS ---
print("Loading models...")
models = {
    "tomato": load_model(os.path.join('models', 'tomato_model.h5')),
    "potato": load_model(os.path.join('models', 'potato_model.h5'))
}
gatekeeper_model = MobileNetV2(weights='imagenet')

# --- LOAD CLASS NAMES ---
class_name_sets = { "tomato": [], "potato": [] }
with open(os.path.join('models', 'tomato_classes.txt'), 'r') as f: class_name_sets["tomato"] = [line.strip() for line in f.readlines()]
with open(os.path.join('models', 'potato_classes.txt'), 'r') as f: class_name_sets["potato"] = [line.strip() for line in f.readlines()]
print("All models loaded.")

# --- HELPER FUNCTIONS ---
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return model.predict(x)

def is_relevant_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = gatekeeper_model.predict(x)
    decoded = decode_predictions(preds, top=10)[0]
    keywords = ['plant', 'leaf', 'flower', 'vegetable', 'fruit', 'tree', 'grass', 'garden', 'agriculture', 'crop', 'tomato', 'potato', 'fungus', 'mold', 'blight', 'spot', 'pest', 'bug', 'beetle', 'insect', 'web']
    for _, label, _ in decoded:
        for k in keywords:
            if k in label.lower(): return True, label
    return False, decoded[0][1]

# --- NEW AI DESCRIPTION FUNCTION ---
def generate_ai_description(image_path, predicted_class):
    """Generates a short visual description using Gemini."""
    try:
        if not chatbot_model: return None
        img = PIL.Image.open(image_path)
        prompt = f"You are an agricultural expert. Look at this plant leaf. The diagnosis is {predicted_class}. Briefly describe the visual symptoms you see that confirm this."
        response = chatbot_model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        print(f"AI Description Error: {e}")
        return None

# --- ROUTES ---  
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        selected_crop = request.form.get('crop_selection')
        
        basepath = os.path.dirname(__file__)
        uploads_folder = os.path.join(basepath, 'static', 'uploads')
        os.makedirs(uploads_folder, exist_ok=True)
        file_path = os.path.join(uploads_folder, f.filename)
        f.save(file_path)

        # 1. Gatekeeper
        is_valid, obj = is_relevant_image(file_path)
        if not is_valid:
            return render_template('index.html', prediction="Image Rejected", details={"display_name": "Not a Plant", "cause": f"AI saw: {obj}", "symptoms": "N/A", "prevention": "N/A", "treatment": "N/A", "severity": "N/A", "scientific_name": "N/A", "conditions": "N/A", "treatment_organic": "N/A", "treatment_chemical": "N/A"}, image_filename=f.filename)

        # 2. Predict
        model = models.get(selected_crop)
        preds = model_predict(file_path, model)
        confidence = np.max(preds) * 100
        
        if confidence < 65:
            return render_template('index.html', prediction="Uncertain", image_filename=f.filename)
        
        class_names = class_name_sets.get(selected_crop)
        pred_name = class_names[np.argmax(preds)]
        details = all_disease_info.get(selected_crop, {}).get(pred_name, {"display_name": pred_name})
        result_name = details.get('display_name')
        
        # 3. Generate AI Description (THE NEW FEATURE)
        ai_desc = generate_ai_description(file_path, result_name)

        # 4. Save to DB
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect('history.db')
        cursor = conn.cursor()
        db_text = f"({selected_crop.capitalize()}) {result_name}"
        cursor.execute("INSERT INTO scans (image_filename, prediction, confidence, scan_time) VALUES (?, ?, ?, ?)",
                       (f.filename, db_text, float(confidence), current_time))
        conn.commit()
        conn.close()
        
        return render_template('index.html', prediction=f"{result_name} ({confidence:.2f}% confidence)", details=details, image_filename=f.filename, ai_description=ai_desc)

    return render_template('index.html', prediction=None)

@app.route('/history')
def history():
    conn = sqlite3.connect('history.db')
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scans ORDER BY scan_time DESC")
    scans = cursor.fetchall()
    conn.close()
    return render_template('history.html', scans=scans)

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('history.db')
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()
    cursor.execute("SELECT prediction, COUNT(*) as count FROM scans GROUP BY prediction")
    data = cursor.fetchall()
    conn.close()
    labels = [row['prediction'] for row in data]
    counts = [row['count'] for row in data]
    return render_template('dashboard.html', labels=labels, data=counts)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    conn = sqlite3.connect('history.db')
    conn.execute("DELETE FROM scans")
    conn.execute("DELETE FROM sqlite_sequence WHERE name='scans'")
    conn.commit()
    conn.close()
    return redirect(url_for('history'))

@app.route('/chat', methods=['POST'])
def chat():
    if not chatbot_model: return json.dumps({"error": "AI not configured"}), 500
    try:
        data = request.json
        response = chatbot_model.generate_content(f"Context: {data.get('context')}. User: {data.get('message')}")
        return json.dumps({"reply": response.text})
    except Exception as e:
        return json.dumps({"error": str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)