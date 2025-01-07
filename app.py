from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertTokenizer
from model import MultimodalClassifier  # Assuming `MultimodalClassifier` is in model.py

# Initialize FastAPI app
app = FastAPI()

# Model and tokenizer setup
model_path = 'Combine_best_model.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = MultimodalClassifier()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define image transformations
transform_test = transforms.Compose([
    transforms.Resize((228, 228)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the tokenizer
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

def predict(image_path: str, tweet_text: str):
    """Predict the label based on image and text."""
    # Process the image
    image = Image.open(image_path).convert('RGB')
    image = transform_test(image).unsqueeze(0).to(device)

    # Process the tweet
    encoding = tokenizer.encode_plus(
        tweet_text,
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

    return label_map[predicted.item()]

@app.post("/predict/")
async def handle_prediction(
    image: UploadFile = File(...), 
    tweet: str = Form(...)
):
    """
    Endpoint to handle predictions.
    Expects:
        - An image file uploaded as 'image'
        - A tweet string as 'tweet'
    Returns:
        - The predicted disaster type as JSON
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    # Save the uploaded image temporarily
    temp_image_path = "temp_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(await image.read())

    try:
        # Predict the result
        prediction = predict(temp_image_path, tweet)
        return JSONResponse(content={"prediction": prediction}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    finally:
        # Remove the temporary file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

@app.get("/")
async def home():
    """Health check endpoint."""
    return {"message": "Welcome to the Disaster Classification API"}

# Run the app using this command in the terminal
# uvicorn app:app --host 0.0.0.0 --port 8000 --reload
