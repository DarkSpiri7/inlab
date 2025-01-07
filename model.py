import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertModel, BertTokenizer

# Path to the saved model weights
model_path = 'Combine_best_model.pt'
num_classes = 6

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultimodalClassifier(nn.Module):
    def __init__(self):
        super(MultimodalClassifier, self).__init__()

        # Image Model (ResNet-50)
        self.image_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_image_features = self.image_model.fc.in_features
        self.image_model.fc = nn.Sequential(
            nn.Linear(num_image_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Text Model (BERT)
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.text_linear = nn.Linear(self.text_model.config.hidden_size, 512)

        # Fusion Layer
        self.fusion_linear = nn.Linear(1024, num_classes)

    def forward(self, image_input, text_input_ids, text_attention_mask):
        # Process image
        image_features = self.image_model(image_input)

        # Process text
        text_outputs = self.text_model(text_input_ids, attention_mask=text_attention_mask)
        pooled_text_output = text_outputs.pooler_output
        text_features = self.text_linear(pooled_text_output)

        # Concatenate image and text features
        multimodal_features = torch.cat((image_features, text_features), dim=1)

        # Fusion layer
        combined_logits = self.fusion_linear(multimodal_features)

        return combined_logits

# Load the model
model = MultimodalClassifier()

# Load model weights
try:
    weights = torch.load(model_path, map_location=device)
    model.load_state_dict(weights)
except Exception as e:
    raise RuntimeError(f"Error loading model weights: {e}")

model.to(device)
model.eval()

# Define transformations for the image
transform_test = transforms.Compose([
    transforms.Resize((228, 228)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def test_model(image_path, tweet):
    """Test the model with a given image and tweet."""
    # Process the image
    image = Image.open(image_path).convert('RGB')
    image = transform_test(image).unsqueeze(0).to(device)

    # Process the tweet
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

    # Map prediction to label
    label_map = {
        0: "non_damage",
        1: "damaged_infrastructure",
        2: "damaged_nature",
        3: "fires",
        4: "flood",
        5: "human_damage"
    }
    return label_map[predicted.item()]

def test_folder(test_images_folder, test_tweets_folder):
    """Test the model on all images and tweets in the specified folders."""
    results = []

    for image_file in os.listdir(test_images_folder):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(test_images_folder, image_file)
            tweet_file = os.path.join(test_tweets_folder, image_file.rsplit('.', 1)[0] + '.txt')

            if os.path.exists(tweet_file):
                try:
                    with open(tweet_file, 'r', encoding='utf-8') as file:
                        tweet = file.read().strip()

                    prediction = test_model(image_path, tweet)
                    results.append({
                        'image': image_file,
                        'tweet_file': os.path.basename(tweet_file),
                        'prediction': prediction
                    })
                except UnicodeDecodeError as e:
                    print(f"Error reading tweet file {tweet_file}: {e}")
            else:
                print(f"Tweet file not found for image: {image_file}")

    return results

# Example usage
if __name__ == "__main__":
    # Update these paths as needed
    test_images_folder = r"E:\Disaster\test_images"
    test_tweets_folder = r"E:\Disaster\test_tweet"

    results = test_folder(test_images_folder, test_tweets_folder)

    for result in results:
        print(f"Image: {result['image']}, Prediction: {result['prediction']}, Tweet File: {result['tweet_file']}")
