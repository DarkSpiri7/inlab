from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize
from transformers import BertTokenizer
from PIL import Image
import os
from model import MultimodalClassifier  # Assuming your model class is in model.py

# Initialize FastAPI app
app = FastAPI()

# Load the model and tokenizer
model_path = 'Combine_best_model.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalClassifier()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define image transformation
transform_test = Compose([
    Resize((228, 228)),
    Grayscale(num_output_channels=3),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Label mapping
label_map = {
    0: "non_damage",
    1: "damaged_infrastructure",
    2: "damaged_nature",
    3: "fires",
    4: "flood",
    5: "human_damage"
}

@app.post("/predict/")
async def predict(image: UploadFile = File(...), tweet: str = Form(...)):
    try:
        # Process the image
        image = Image.open(image.file).convert('RGB')
        image = transform_test(image).unsqueeze(0).to(device)

        # Process the text
        encoding = tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=300,
            truncation=True,
            return_tensors='pt',
            padding='max_length'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Predict
        with torch.no_grad():
            output = model(image, input_ids, attention_mask)
            _, predicted = torch.max(output, 1)

        # Return result
        return {"prediction": label_map[predicted.item()]}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
