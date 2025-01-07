import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import BertModel, BertTokenizer

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
        self.fusion_linear = nn.Linear(1024, 6)  # Assuming 6 classes as an example

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

# Quantize the model
def quantize_model(model):
    """Quantize the model using dynamic quantization."""
    model.eval()  # Ensure the model is in evaluation mode
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear},  # Apply quantization to the Linear layers
        dtype=torch.qint8  # Using 8-bit integer quantization
    )
    return quantized_model

# Load and quantize model weights
def load_quantized_model(model_path):
    model = MultimodalClassifier()
    weights = torch.load(model_path, map_location=torch.device('cpu'))  # Assuming CPU for inference
    model.load_state_dict(weights)
    quantized_model = quantize_model(model)
    return quantized_model
