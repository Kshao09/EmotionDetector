# Emotion Detection (EfficientNet-B0 + FastAPI + MediaPipe)

Real-time facial emotion recognition web app.  
Upload an image or use the live camera in the browser, and the server predicts one of **7 emotions** using a fine-tuned **EfficientNet-B0** model. The app uses **MediaPipe** face detection to crop the face before classification for more stable predictions.

## Demo Features
- ✅ FastAPI backend inference API (`/predict`)
- ✅ Web UI:
  - Image upload prediction
  - Live camera prediction (webcam)
  - Face bounding box overlay
  - Adjustable prediction interval + smoothing
- ✅ MediaPipe face detection + face crop before model inference
- ✅ PyTorch EfficientNet-B0 classifier (7 classes)

---

## Dataset Used: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer

## Emotion Classes
This project predicts 7 emotions:

- angry
- disgusted
- fearful
- happy
- neutral
- sad
- surprised

## Tech Stack
- **Model:** PyTorch `torchvision.models.efficientnet_b0`
- **Backend:** FastAPI + Uvicorn
- **Face Detection:** MediaPipe `face_detection`
- **Frontend:** Simple HTML/JS (served from FastAPI)
- **Preprocessing:** Torchvision transforms + ImageNet normalization

---

## Project Structure (Typical)
├── main.py
├── emotion_efficientnet_b0.pth
├── class_names.json
├── requirements.txt
└── (optional) notebooks/
