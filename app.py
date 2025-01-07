from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from PIL import Image
from transformers import BertTokenizer
from model import load_quantized_model  # Import the quantization function from model.py

# Define FastAPI app
app = FastAPI()

# Load the quantized model once at startup
model_path = 'Combine_best_model.pt'  # Path to your model weights
model = load_quantized_model(model_path)  # Load the quantized model

# Load the tokenizer for BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define transformation for image processing
transform_test = transforms.Compose([
    transforms.Resize((228, 228)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define request body model
class PredictionRequest(BaseModel):
    image: str  # Base64 encoded image string
    tweet: str   # Tweet text

# Define prediction endpoint
@app.post("/predict/")
async def predict(request: PredictionRequest):
    try:
        # Decode the base64 image
        import base64
        from io import BytesIO
        img_data = base64.b64decode(request.image)
        image = Image.open(BytesIO(img_data)).convert('RGB')
        image = transform_test(image).unsqueeze(0)  # Transform image
        
        # Process the tweet
        encoding = tokenizer.encode_plus(
            request.tweet,
            add_special_tokens=True,
            max_length=300,
            truncation=True,
            return_tensors='pt',
            padding='max_length'
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Perform prediction using the quantized model
        with torch.no_grad():
            output = model(image, input_ids, attention_mask)
            _, predicted = torch.max(output, 1)

        # Map prediction to label
        label_map = {
            0: "non_damage",
            1: "damaged_infrastructure",
            2: "damaged_nature",
            3: "fires",
            4: "flood",
            5: "human_damage"
        }
        prediction = label_map[predicted.item()]

        return {"prediction": prediction}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app (FastAPI)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
