
# Load model directly
import torch
from PIL import Image
import torch.nn.functional as F
from transformers import AutoModelForImageClassification, ViTImageProcessor

img = Image.open("C:\\Users\\user\\Downloads\\wolverine_death.jpg")
model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
processor = ViTImageProcessor.from_pretrained('Falconsai/nsfw_image_detection')

def nsfw_falcon_detect(images):
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    nsfw_probs = probabilities [:,1].tolist()

    return nsfw_probs

print(nsfw_falcon_detect([img]))
