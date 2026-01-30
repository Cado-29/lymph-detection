import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

MODEL_PATH = "model.pth"

CLASS_NAMES = ['Fake', 'Real'] 

app = FastAPI(title="Lymphoma Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model():
    print("⏳ Loading EfficientNetV2-S model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Recreate the exact architecture used in training
    model = models.efficientnet_v2_s(weights=None)
    
    # Recreate the modified classifier head
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    
    # Load the state dictionary
    try:
        # map_location ensures it loads on CPU if you trained on GPU but deploy on CPU
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval() # Set to evaluation mode (freezes BatchNorm/Dropout)
        print(f"✅ Model loaded successfully on {device}!")
        return model, device
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise RuntimeError("Model loading failed")

# Load model globally so it stays in memory
model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {"message": "Lymphoma Detection API is Running"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # 1. Validate File Type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")
    
    try:
        # 2. Read and Transform Image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0) # Add batch dimension -> [1, 3, 224, 224]
        tensor = tensor.to(device)

        # 3. Inference
        with torch.no_grad():
            outputs = model(tensor)
            # Apply Softmax to get probabilities (0 to 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top prediction
            confidence, predicted_class_idx = torch.max(probs, 1)
            
            # Convert to Python types
            confidence_score = confidence.item()
            predicted_label = CLASS_NAMES[predicted_class_idx.item()]
            
            # Get probability for "Fake" specifically (useful for frontend gauges)
            fake_probability = probs[0][0].item()
            real_probability = probs[0][1].item()

        return {
            "filename": file.filename,
            "prediction": predicted_label,
            "confidence": round(confidence_score * 100, 2),
            "probabilities": {
                "fake": round(fake_probability * 100, 2),
                "real": round(real_probability * 100, 2)
            }
        }

    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during processing.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)