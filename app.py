from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import torch
from torchvision import models, transforms
from PIL import Image
import io, json
import numpy as np
import mediapipe as mp
from fastapi.responses import HTMLResponse

app = FastAPI()

# ---- load class names ----
with open("class_names.json", "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)

# ---- load model (same architecture) ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, num_classes)

state = torch.load("emotion_efficientnet_b0.pth", map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def crop_face_pil(pil_img, pad=0.25):
    """
    Returns (cropped_face_pil, bbox_norm)
    bbox_norm = {"xmin":..., "ymin":..., "xmax":..., "ymax":...} in [0,1] relative coords (after padding).
    If no face found, returns (original_img, None)
    """
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]

    results = mp_face.process(img)
    if not results.detections:
        return pil_img, None

    # pick the largest face
    best = None
    best_area = 0
    for det in results.detections:
        box = det.location_data.relative_bounding_box
        x1 = box.xmin * w
        y1 = box.ymin * h
        bw = box.width * w
        bh = box.height * h
        area = bw * bh
        if area > best_area:
            best_area = area
            best = (x1, y1, bw, bh)

    x1, y1, bw, bh = best

    # padding
    px = bw * pad
    py = bh * pad
    x1p = max(0, x1 - px)
    y1p = max(0, y1 - py)
    x2p = min(w, x1 + bw + 2 * px)
    y2p = min(h, y1 + bh + 2 * py)

    x1i, y1i, x2i, y2i = int(x1p), int(y1p), int(x2p), int(y2p)
    face = img[y1i:y2i, x1i:x2i]

    bbox_norm = {
        "xmin": x1p / w,
        "ymin": y1p / h,
        "xmax": x2p / w,
        "ymax": y2p / h,
    }
    return Image.fromarray(face), bbox_norm


# ---- same preprocessing as test ----
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict(img: Image.Image):
    img = img.convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        pred = int(torch.argmax(logits, dim=1).item())
    return pred, probs

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    b = await file.read()
    img = Image.open(io.BytesIO(b))

    face_img, bbox = crop_face_pil(img)

    pred, probs = predict(face_img)
    emotion = class_names[pred]
    conf = float(probs[pred])

    return {
        "emotion": emotion,
        "confidence": conf,
        "bbox": bbox,            # None if no face
        # "probs": probs,        # optional, if you want client-side probability smoothing
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <head>
        <style>
          body { font-family: Arial; max-width: 980px; margin: 40px auto; }
          #videoBox { position: relative; width: 460px; }
          video { width: 460px; border-radius: 12px; border: 1px solid #ddd; background:#000; }
          canvas#overlay { position: absolute; left: 0; top: 0; width: 460px; height: auto; pointer-events:none; }
          #result { margin-top: 14px; font-size: 18px; font-weight: 600; }
          button { padding: 8px 12px; margin-right: 8px; }
          .row { margin-top: 12px; display: flex; gap: 18px; align-items: center; flex-wrap: wrap; }
          .small { font-size: 13px; color: #666; line-height: 1.4; }
        </style>
      </head>
      <body>
        <h1>Emotion Detection (Live Camera)</h1>

        <div class="row">
          <button onclick="startCamera()">Start</button>
          <button onclick="stopCamera()">Stop</button>

          <label>Predict every:
            <select id="interval">
              <option value="500">0.5s</option>
              <option value="800" selected>0.8s</option>
              <option value="1200">1.2s</option>
              <option value="2000">2.0s</option>
            </select>
          </label>

          <label>Smoothing window:
            <select id="smoothN">
              <option value="1">1 (off)</option>
              <option value="3">3</option>
              <option value="5" selected>5</option>
              <option value="7">7</option>
            </select>
          </label>
        </div>

        <div class="small">
          Camera works on <b>http://localhost</b> or <b>https</b>. If you host remotely without https, browsers often block webcam access.
        </div>

        <div class="row" style="align-items:flex-start;">
          <div id="videoBox">
            <video id="video" autoplay playsinline muted></video>
            <canvas id="overlay"></canvas>
          </div>

          <div style="flex:1;">
            <div id="result">Idle</div>
            <div id="status" class="small"></div>
          </div>
        </div>

        <canvas id="capture" style="display:none;"></canvas>

        <script>
          const video = document.getElementById("video");
          const overlay = document.getElementById("overlay");
          const capture = document.getElementById("capture");
          const result = document.getElementById("result");
          const statusEl = document.getElementById("status");
          const intervalSel = document.getElementById("interval");
          const smoothSel = document.getElementById("smoothN");

          let stream = null;
          let timer = null;
          let inFlight = false;

          // keep last N predictions for smoothing
          let history = []; // {emotion, confidence}

          const emotionColor = (emotion) => ({
            angry: "#ff3b30",
            disgusted: "#34c759",
            fearful: "#af52de",
            happy: "#ffcc00",
            neutral: "#0a84ff",
            sad: "#5e5ce6",
            surprised: "#ff9f0a"
          }[emotion] || "#00c7ff");

          function clearOverlay() {
            const ctx = overlay.getContext("2d");
            ctx.clearRect(0, 0, overlay.width, overlay.height);
          }

          function drawBox(bbox, label, color) {
            const ctx = overlay.getContext("2d");
            ctx.clearRect(0, 0, overlay.width, overlay.height);

            if (!bbox) return;

            // overlay canvas is sized to match the *displayed* video pixels
            const ow = overlay.width;
            const oh = overlay.height;

            const x1 = bbox.xmin * ow;
            const y1 = bbox.ymin * oh;
            const x2 = bbox.xmax * ow;
            const y2 = bbox.ymax * oh;

            ctx.lineWidth = 4;
            ctx.strokeStyle = color;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            // label background
            const text = label;
            ctx.font = "18px Arial";
            const tw = ctx.measureText(text).width;
            const pad = 6;
            const bx = x1;
            const by = Math.max(0, y1 - 28);

            ctx.fillStyle = "rgba(0,0,0,0.6)";
            ctx.fillRect(bx, by, tw + pad*2, 26);

            ctx.fillStyle = "#fff";
            ctx.fillText(text, bx + pad, by + 19);
          }

          function modeEmotion(items) {
            const counts = {};
            for (const it of items) counts[it.emotion] = (counts[it.emotion] || 0) + 1;
            let best = null, bestCount = -1;
            for (const [k,v] of Object.entries(counts)) {
              if (v > bestCount) { best = k; bestCount = v; }
            }
            return best;
          }

          async function startCamera() {
            try {
              statusEl.textContent = "Requesting camera permission...";
              stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
              video.srcObject = stream;
              await video.play();

              // IMPORTANT: set overlay resolution to match displayed video resolution
              // We'll sync sizes after the video metadata loads
              syncOverlaySize();

              statusEl.textContent = "Camera started.";
              startLoop();
            } catch (e) {
              statusEl.textContent = "Camera error: " + e.message;
            }
          }

          function syncOverlaySize() {
            // wait a tick for layout
            setTimeout(() => {
              const rect = video.getBoundingClientRect();
              overlay.width = Math.round(rect.width);
              overlay.height = Math.round(rect.height);
              clearOverlay();
            }, 100);
          }

          window.addEventListener("resize", () => {
            if (stream) syncOverlaySize();
          });

          function stopCamera() {
            if (timer) clearInterval(timer);
            timer = null;
            inFlight = false;
            history = [];
            clearOverlay();

            if (stream) stream.getTracks().forEach(t => t.stop());
            stream = null;
            video.srcObject = null;

            result.textContent = "Stopped";
            statusEl.textContent = "";
          }

          function startLoop() {
            if (timer) clearInterval(timer);
            const intervalMs = parseInt(intervalSel.value || "800", 10);

            timer = setInterval(() => {
              if (!stream) return;
              if (inFlight) return;
              captureAndPredict();
            }, intervalMs);
          }

          intervalSel.addEventListener("change", () => {
            if (stream) startLoop();
          });

          async function captureAndPredict() {
            if (!video.videoWidth || !video.videoHeight) return;

            // capture at native video resolution (best for backend face detection)
            capture.width = video.videoWidth;
            capture.height = video.videoHeight;
            const cctx = capture.getContext("2d");
            cctx.drawImage(video, 0, 0, capture.width, capture.height);

            inFlight = true;

            capture.toBlob(async (blob) => {
              try {
                const fd = new FormData();
                fd.append("file", blob, "frame.jpg");

                const res = await fetch("/predict", { method: "POST", body: fd });
                const data = await res.json();

                if (!data.bbox) {
                  result.textContent = "No face detected";
                  statusEl.textContent = "Try better lighting, face closer to camera, or adjust angle.";
                  clearOverlay();
                  history = [];
                  return;
                }

                const emotion = data.emotion;
                const pct = (data.confidence * 100).toFixed(1);

                // smoothing
                const N = parseInt(smoothSel.value || "5", 10);
                history.push({ emotion, confidence: data.confidence });
                if (history.length > N) history.shift();

                const smEmotion = (N <= 1) ? emotion : modeEmotion(history);
                const color = emotionColor(smEmotion);

                result.innerHTML = `Result: <b>${smEmotion}</b> (${pct}%)`;
                statusEl.textContent = "";

                // bbox is normalized; draw on overlay (display-sized)
                drawBox(data.bbox, `${smEmotion} (${pct}%)`, color);

              } catch (e) {
                result.textContent = "Error";
                statusEl.textContent = "Predict error: " + e.message;
                clearOverlay();
              } finally {
                inFlight = false;
              }
            }, "image/jpeg", 0.85);
          }
        </script>
      </body>
    </html>
    """
