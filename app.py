from flask import Flask, render_template, request
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from googletrans import Translator

app = Flask(__name__)
translator = Translator()

# Load model
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 15)
model.load_state_dict(torch.load('model/model_state_dict.pt', map_location=torch.device('cpu')))
model.eval()

# Preprocessing
preprocess = weights.transforms()

# Class labels and details
class_labels = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Late_blight',
    'Tomato___healthy'
]

disease_info = {
    "Apple_scab": {
        "desc": "A fungal disease causing dark, scabby lesions on leaves and fruits.",
        "steps": "Remove fallen leaves, use fungicides like Mancozeb.",
        "fertilizers": "Neem oil spray, copper-based fungicide"
    },
    "Black_rot": {
        "desc": "Caused by fungus; results in leaf spots and fruit rots.",
        "steps": "Remove infected parts, apply fungicide.",
        "fertilizers": "Bordeaux mixture, copper oxychloride"
    },
    "Cedar_apple_rust": {
        "desc": "A fungal disease affecting apples and cedar trees.",
        "steps": "Remove nearby cedars, use sulfur sprays.",
        "fertilizers": "Fungicides with myclobutanil"
    },
    "Cercospora_leaf_spot": {
        "desc": "Fungal disease causing brown lesions on corn leaves.",
        "steps": "Rotate crops, use resistant varieties.",
        "fertilizers": "Azoxystrobin-based fungicides"
    },
    "Common_rust": {
        "desc": "Rust pustules on corn caused by Puccinia sorghi.",
        "steps": "Plant resistant hybrids, use fungicides.",
        "fertilizers": "Propiconazole sprays"
    },
    "Northern_Leaf_Blight": {
        "desc": "Caused by Exserohilum turcicum, appears as cigar-shaped lesions.",
        "steps": "Use resistant hybrids, remove infected debris.",
        "fertilizers": "Strobilurin group fungicides"
    },
    "Black_rot_grape": {
        "desc": "A grape disease causing circular black lesions.",
        "steps": "Prune and destroy infected parts, use fungicides.",
        "fertilizers": "Ziram, captan fungicides"
    },
    "Esca": {
        "desc": "Trunk disease in grapes causing leaf stripes and wilting.",
        "steps": "Avoid water stress, prune infected vines.",
        "fertilizers": "No specific fertilizer, manage vineyard health."
    },
    "Leaf_blight": {
        "desc": "Caused by fungal spores, brown necrotic spots on grape leaves.",
        "steps": "Use healthy plants, copper-based sprays.",
        "fertilizers": "Copper hydroxide"
    },
    "Bacterial_spot": {
        "desc": "Tomato disease caused by Xanthomonas; leaf spots and fruit damage.",
        "steps": "Use disease-free seeds, copper sprays.",
        "fertilizers": "Copper sulfate, streptomycin"
    },
    "Late_blight": {
        "desc": "Severe disease caused by Phytophthora infestans, affects leaves and fruit.",
        "steps": "Destroy infected plants, apply fungicide regularly.",
        "fertilizers": "Chlorothalonil, mancozeb"
    }
}

def transform_image(image_file):
    image = Image.open(image_file).convert('RGB')
    return preprocess(image).unsqueeze(0)

def get_prediction(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    label = class_labels[predicted.item()]
    return label

def parse_prediction(label):
    plant, condition = label.split("___")
    is_healthy = "healthy" in condition.lower()
    if is_healthy:
        return plant, "Healthy", "No action needed.", "-"
    else:
        key = condition.replace(" ", "_")
        key = f"{key}" if plant != "Grape" else f"{key}_grape"
        info = disease_info.get(key, {
            "desc": "No description available.",
            "steps": "No prevention steps found.",
            "fertilizers": "-"
        })
        return plant, condition.replace("_", " "), info["desc"], info["steps"], info["fertilizers"]

def translate_text(text, lang):
    if lang == "en":
        return text
    try:
        translated = translator.translate(text, dest=lang)
        return translated.text
    except Exception as e:
        print("Translation failed:", e)
        return text

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('index.html', prediction="No file uploaded.")

    file = request.files['file']
    language = request.form.get("language", "en")

    image_tensor = transform_image(file)
    label = get_prediction(image_tensor)
    plant, status, description, steps, fertilizers = parse_prediction(label)

    # Translate output
    plant = translate_text(plant, language)
    status = translate_text(status, language)
    description = translate_text(description, language)
    steps = translate_text(steps, language)
    fertilizers = translate_text(fertilizers, language)

    return render_template('index.html',
                           plant=plant,
                           status=status,
                           description=description,
                           steps=steps,
                           fertilizers=fertilizers,
                           prediction=f"{plant} - {status}")

if __name__ == '__main__':
    app.run(debug=True)
