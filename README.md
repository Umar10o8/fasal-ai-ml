# Rice Crop Advisory System — Integration Guide

> **Current deployment scope:** this repository currently includes the fertilizer
> recommendation model and the `POST /predict/fertilizer` endpoint only. Disease
> and pest model files/endpoints described in the integration notes below are not
> part of the current running app.

## Deploy the FastAPI service on Render

This repository includes a `render.yaml` Blueprint for deploying the FastAPI
fertilizer service as a Render web service. The included `.python-version` pins
the runtime to Python 3.13.

1. Push this repository to GitHub with `app/models/fertilizer_rf_model.pkl`
   included. Do not push `.env` or a virtual environment.
2. In Render, choose **New → Blueprint**, connect this repository, and apply the
   `render.yaml` configuration.
3. Wait for the service to deploy. Open `https://<your-service>.onrender.com/health`
   and confirm the response has `"status":"ok"` and
   `"fertilizer_model_loaded":true`.
4. In the backend's Vercel environment variables, set
   `FERTILIZER_ML_SERVICE_URL` to `https://<your-service>.onrender.com` (no trailing
   slash), then redeploy the backend.

The free Render service can spin down while idle, so the first request can be slow.
The current backend ML request timeout is 15 seconds; wake the service by opening
`/health`, wait for the model to load, and retry a recommendation if the first
request times out.

### Disease Detection · Pest Identification · Fertilizer Recommendation

This README explains how to take the three trained models you already have —

| Module | Model type | Artifact |
|---|---|---|
| Crop Disease Detection | YOLO | `disease_yolo.pt` |
| Pest Identification (RicePest-30) | YOLO | `pest_yolo.pt` |
| Fertilizer Recommendation | Random Forest (sklearn Pipeline) | `fertilizer_rf_model.pkl` |

— and wire them up to your **Node.js backend**, which is then consumed by a **web page** and a **mobile app**.

> **Important**: Node.js cannot load a `.pt` (YOLO) or `.pkl` (sklearn) file natively — there's no equivalent runtime for these in JavaScript. So the models still run inside a small **Python inference microservice** (FastAPI), and your **Node.js app stays the main backend** — it owns auth, business logic, and the public API — and simply forwards ML requests to that internal Python service, then relays the response back to the client. Neither your web page nor your mobile app talk to Python directly; they only ever talk to Node.

---

## 1. Architecture Overview

```
                        ┌─────────────────────────┐
                        │      Web Page (JS)      │
                        │   React / Vanilla HTML  │
                        └───────────┬─────────────┘
                                    │ HTTPS (JSON / multipart)
                                    ▼
                  ┌─────────────────────────┐
                  │     Mobile App           │
                  │  Flutter / React Native  │
                  └───────────┬─────────────┘
                              │ HTTPS (JSON / multipart)
                              ▼
┌───────────────────────────────────────────────────────────────┐
│              Node.js Backend (Express) — public API            │
│                                                                 │
│   Owns: auth, business logic, DB, rate limiting, routing        │
│   POST /api/disease      ─┐                                    │
│   POST /api/pest          ├─► forwards to Python service        │
│   POST /api/fertilizer   ─┘                                    │
└───────────────────────────┬────────────────────────────────────┘
                             │ internal HTTP call (localhost / private network)
                             ▼
┌───────────────────────────────────────────────────────────────┐
│           Python Inference Microservice (FastAPI)              │
│                                                                 │
│   POST /predict/disease     → loads disease_yolo.pt            │
│   POST /predict/pest        → loads pest_yolo.pt                │
│   POST /predict/fertilizer  → loads fertilizer_rf_model.pkl     │
│                                                                 │
│   Models are loaded ONCE at server startup, kept in memory      │
└───────────────────────────────────────────────────────────────┘
```

Node is the only thing the frontend/app ever calls. The Python service is internal — not exposed publicly, only reachable from Node (e.g. over `localhost` if on the same machine, or a private network/VPC if split across servers).

---

## 2. Python Inference Microservice (FastAPI)

This is the internal service that actually runs the three models. Your Node backend calls this — the web page and mobile app never see it directly.

### 2.1 Folder structure

```
rice-advisory/
├── ml-service/                  # Python — internal, not public-facing
│   ├── app/
│   │   ├── main.py
│   │   ├── models/
│   │   │   ├── disease_yolo.pt
│   │   │   ├── pest_yolo.pt
│   │   │   └── fertilizer_rf_model.pkl
│   │   └── schemas.py
│   ├── requirements.txt
│   └── Dockerfile
│
└── node-backend/                # Node.js — public-facing API
    ├── server.js
    ├── routes/
    │   └── ml.js
    ├── package.json
    └── Dockerfile
```

### 2.2 `requirements.txt`

Versions matter here more than usual: `scikit-learn`, `joblib`, `pandas`, and `numpy` are pinned to the exact versions the `fertilizer_rf_model.pkl` file was trained and saved with — a newer/older sklearn can fail to unpickle the model or silently behave differently.

```
fastapi==0.139.2
uvicorn[standard]==0.51.0
python-multipart==0.0.32
ultralytics==8.3.253
scikit-learn==1.8.0
joblib==1.5.3
pandas==3.0.2
numpy==2.4.4
Pillow==11.3.0
```

### 2.3 `app/schemas.py` — request body for the fertilizer endpoint

```python
from pydantic import BaseModel

class FertilizerInput(BaseModel):
    Soil_pH: float
    Soil_Moisture: float
    Organic_Carbon: float
    Electrical_Conductivity: float
    Nitrogen_Level: int
    Phosphorus_Level: int
    Potassium_Level: int
    Temperature: float
    Humidity: float
    Rainfall: float
    Fertilizer_Used_Last_Season: float
    Yield_Last_Season: float
    Soil_Type: str
    Crop_Type: str
    Crop_Growth_Stage: str
    Season: str
    Irrigation_Type: str
    Previous_Crop: str
    Region: str
```

### 2.4 `app/main.py` — the three endpoints

```python
import io
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

from app.schemas import FertilizerInput

app = FastAPI(title="Rice Advisory ML Service (internal)")

# This service is internal — only the Node.js backend should ever call it,
# never the web page or mobile app directly. Lock CORS down to Node's own origin,
# or skip CORS entirely if both run on the same private network (recommended).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],   # your Node backend's address
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load all models ONCE at startup, not per-request ----
disease_model = YOLO("app/models/disease_yolo.pt")
pest_model = YOLO("app/models/pest_yolo.pt")
fertilizer_model = joblib.load("app/models/fertilizer_rf_model.pkl")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/disease")
async def predict_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = disease_model.predict(image, conf=0.4)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": disease_model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 3),
                "bbox": [round(v, 1) for v in box.xyxy[0].tolist()]
            })

    return {"detections": detections}


@app.post("/predict/pest")
async def predict_pest(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = pest_model.predict(image, conf=0.4)
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": pest_model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 3),
                "bbox": [round(v, 1) for v in box.xyxy[0].tolist()]
            })

    return {"detections": detections}


@app.post("/predict/fertilizer")
def predict_fertilizer(data: FertilizerInput):
    input_df = pd.DataFrame([data.model_dump()])

    prediction = fertilizer_model.predict(input_df)[0]
    proba = fertilizer_model.predict_proba(input_df)[0]
    classes = fertilizer_model.named_steps["classifier"].classes_

    top3_idx = proba.argsort()[::-1][:3]
    top3 = [{"fertilizer": classes[i], "probability": round(float(proba[i]), 3)} for i in top3_idx]

    return {"recommended_fertilizer": prediction, "top3": top3}
```

### 2.5 Run it locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs are then auto-available at `http://localhost:8000/docs` — useful for testing each endpoint from the browser before wiring up the frontend.

### 2.6 Quick test with curl

```bash
# Fertilizer
curl -X POST http://localhost:8000/predict/fertilizer \
  -H "Content-Type: application/json" \
  -d '{
    "Soil_pH": 6.2, "Soil_Moisture": 45.0, "Organic_Carbon": 0.8,
    "Electrical_Conductivity": 1.1, "Nitrogen_Level": 55, "Phosphorus_Level": 30,
    "Potassium_Level": 40, "Temperature": 28.5, "Humidity": 78.0, "Rainfall": 1500.0,
    "Fertilizer_Used_Last_Season": 150.0, "Yield_Last_Season": 4.2,
    "Soil_Type": "Clay", "Crop_Type": "Rice", "Crop_Growth_Stage": "Vegetative",
    "Season": "Kharif", "Irrigation_Type": "Canal", "Previous_Crop": "Wheat", "Region": "South"
  }'

# Disease detection (image upload)
curl -X POST http://localhost:8000/predict/disease \
  -F "file=@sample_leaf.jpg"

# Pest identification (image upload)
curl -X POST http://localhost:8000/predict/pest \
  -F "file=@sample_pest.jpg"
```

---

## 3. Node.js Backend (Express) — the public API

This is what your web page and mobile app actually call. It forwards each request to the Python microservice and relays the result back — no ML code lives here, just routing/proxying plus whatever auth and business logic your app needs.

### 3.1 `package.json`

```json
{
  "name": "rice-advisory-node-backend",
  "version": "1.0.0",
  "main": "server.js",
  "type": "commonjs",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js"
  },
  "dependencies": {
    "express": "^4.21.2",
    "axios": "^1.7.9",
    "multer": "^1.4.5-lts.1",
    "form-data": "^4.0.1",
    "cors": "^2.8.5",
    "dotenv": "^16.4.7"
  },
  "devDependencies": {
    "nodemon": "^3.1.9"
  }
}
```

### 3.2 `.env`

```
ML_SERVICE_URL=http://localhost:8000
PORT=3000
```

### 3.3 `routes/ml.js`

```javascript
const express = require("express");
const axios = require("axios");
const multer = require("multer");
const FormData = require("form-data");

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage() });

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";

// ---- Fertilizer recommendation ----
router.post("/fertilizer", async (req, res) => {
  try {
    const response = await axios.post(`${ML_SERVICE_URL}/predict/fertilizer`, req.body);
    res.json(response.data);
  } catch (err) {
    const status = err.response?.status || 500;
    res.status(status).json({ error: err.response?.data?.detail || "Fertilizer service error" });
  }
});

// ---- Disease detection (image upload) ----
router.post("/disease", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No image uploaded" });

  try {
    const form = new FormData();
    form.append("file", req.file.buffer, { filename: req.file.originalname, contentType: req.file.mimetype });

    const response = await axios.post(`${ML_SERVICE_URL}/predict/disease`, form, {
      headers: form.getHeaders(),
    });
    res.json(response.data);
  } catch (err) {
    const status = err.response?.status || 500;
    res.status(status).json({ error: err.response?.data?.detail || "Disease service error" });
  }
});

// ---- Pest identification (image upload) ----
router.post("/pest", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No image uploaded" });

  try {
    const form = new FormData();
    form.append("file", req.file.buffer, { filename: req.file.originalname, contentType: req.file.mimetype });

    const response = await axios.post(`${ML_SERVICE_URL}/predict/pest`, form, {
      headers: form.getHeaders(),
    });
    res.json(response.data);
  } catch (err) {
    const status = err.response?.status || 500;
    res.status(status).json({ error: err.response?.data?.detail || "Pest service error" });
  }
});

module.exports = router;
```

### 3.4 `server.js`

```javascript
require("dotenv").config();
const express = require("express");
const cors = require("cors");
const mlRoutes = require("./routes/ml");

const app = express();
app.use(cors());              // configure allowed origins for your actual web/app domains
app.use(express.json());

app.get("/health", (req, res) => res.json({ status: "ok" }));
app.use("/api", mlRoutes);     // exposes /api/fertilizer, /api/disease, /api/pest

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Node backend running on port ${PORT}`));
```

### 3.5 Run it locally

```bash
npm install
npm run dev
```

Make sure the Python microservice (Section 2) is running on port 8000 first, since Node forwards to it.

### 3.6 Quick test with curl (now hitting Node, not Python directly)

```bash
# Fertilizer
curl -X POST http://localhost:3000/api/fertilizer \
  -H "Content-Type: application/json" \
  -d '{
    "Soil_pH": 6.2, "Soil_Moisture": 45.0, "Organic_Carbon": 0.8,
    "Electrical_Conductivity": 1.1, "Nitrogen_Level": 55, "Phosphorus_Level": 30,
    "Potassium_Level": 40, "Temperature": 28.5, "Humidity": 78.0, "Rainfall": 1500.0,
    "Fertilizer_Used_Last_Season": 150.0, "Yield_Last_Season": 4.2,
    "Soil_Type": "Clay", "Crop_Type": "Rice", "Crop_Growth_Stage": "Vegetative",
    "Season": "Kharif", "Irrigation_Type": "Canal", "Previous_Crop": "Wheat", "Region": "South"
  }'

# Disease detection (image upload)
curl -X POST http://localhost:3000/api/disease -F "file=@sample_leaf.jpg"

# Pest identification (image upload)
curl -X POST http://localhost:3000/api/pest -F "file=@sample_pest.jpg"
```

---

## 4. API Contract (what the web page and mobile app need to know)

These are the only URLs your frontend/app ever calls — always via Node, never the Python service directly.

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/api/disease` | POST | `multipart/form-data`, field `file` (image) | `{ "detections": [{class, confidence, bbox}] }` |
| `/api/pest` | POST | `multipart/form-data`, field `file` (image) | `{ "detections": [{class, confidence, bbox}] }` |
| `/api/fertilizer` | POST | `application/json` — 18 fields (see `FertilizerInput`) | `{ "recommended_fertilizer": str, "top3": [{fertilizer, probability}] }` |

Both the web page and the mobile app hit these exact same three URLs — nothing platform-specific about the API itself.

---

## 5. Web Page Integration (JavaScript)

### 5.1 Fertilizer recommendation (JSON form)

```javascript
async function getFertilizerRecommendation(formData) {
  const response = await fetch("https://your-node-backend.com/api/fertilizer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(formData)
  });

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return await response.json();
}

// Example usage after a farmer fills a form:
const formData = {
  Soil_pH: 6.2, Soil_Moisture: 45.0, Organic_Carbon: 0.8,
  Electrical_Conductivity: 1.1, Nitrogen_Level: 55, Phosphorus_Level: 30,
  Potassium_Level: 40, Temperature: 28.5, Humidity: 78.0, Rainfall: 1500.0,
  Fertilizer_Used_Last_Season: 150.0, Yield_Last_Season: 4.2,
  Soil_Type: "Clay", Crop_Type: "Rice", Crop_Growth_Stage: "Vegetative",
  Season: "Kharif", Irrigation_Type: "Canal", Previous_Crop: "Wheat", Region: "South"
};

getFertilizerRecommendation(formData).then(result => {
  console.log(result.recommended_fertilizer, result.top3);
});
```

### 5.2 Disease / pest detection (image upload)

```javascript
async function detectDisease(imageFile) {
  const form = new FormData();
  form.append("file", imageFile); // imageFile comes from <input type="file">

  const response = await fetch("https://your-node-backend.com/api/disease", {
    method: "POST",
    body: form
  });
  return await response.json();
}

// Wiring it to an <input id="leafImage" type="file"> element:
document.getElementById("leafImage").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  const result = await detectDisease(file);
  console.log(result.detections);
});
```

Drawing boxes on the image (once you have `bbox` + `class` back) is a canvas overlay — happy to build that as a separate artifact if useful.

---

## 6. Mobile App Integration

### 6.1 Flutter (Dart)

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>> getFertilizerRecommendation(Map<String, dynamic> data) async {
  final response = await http.post(
    Uri.parse('https://your-node-backend.com/api/fertilizer'),
    headers: {'Content-Type': 'application/json'},
    body: jsonEncode(data),
  );

  if (response.statusCode != 200) {
    throw Exception('Failed to get recommendation');
  }
  return jsonDecode(response.body);
}

Future<Map<String, dynamic>> detectDisease(String imagePath) async {
  final uri = Uri.parse('https://your-node-backend.com/api/disease');
  final request = http.MultipartRequest('POST', uri)
    ..files.add(await http.MultipartFile.fromPath('file', imagePath));

  final streamedResponse = await request.send();
  final response = await http.Response.fromStream(streamedResponse);
  return jsonDecode(response.body);
}
```

### 6.2 React Native (JavaScript)

```javascript
// Fertilizer
async function getFertilizerRecommendation(data) {
  const response = await fetch("https://your-node-backend.com/api/fertilizer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return await response.json();
}

// Disease detection — uploading a photo taken with the camera
async function detectDisease(photoUri) {
  const form = new FormData();
  form.append("file", {
    uri: photoUri,
    name: "leaf.jpg",
    type: "image/jpeg",
  });

  const response = await fetch("https://your-node-backend.com/api/disease", {
    method: "POST",
    body: form,
  });
  return await response.json();
}
```

The pest endpoint follows the exact same pattern as disease detection on both platforms — just point at `/api/pest`.

---

## 7. Deployment

### 7.1 Python microservice `Dockerfile` (`ml-service/Dockerfile`)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 7.2 Node backend `Dockerfile` (`node-backend/Dockerfile`)

```dockerfile
FROM node:20-slim

WORKDIR /app
COPY package*.json ./
RUN npm install --production

COPY . .

EXPOSE 3000
CMD ["node", "server.js"]
```

### 7.3 `docker-compose.yml` (runs both together)

```yaml
version: "3.8"
services:
  ml-service:
    build: ./ml-service
    ports:
      - "8000:8000"

  node-backend:
    build: ./node-backend
    ports:
      - "3000:3000"
    environment:
      - ML_SERVICE_URL=http://ml-service:8000
    depends_on:
      - ml-service
```

```bash
docker compose up --build
```

### 7.4 Where to host it
- Both containers can run on the same VM, or `ml-service` on a GPU instance and `node-backend` on a small CPU instance, connected over a private network — a GPU is only worth it if YOLO inference is too slow on CPU for your expected image volume.
- Point your web page and mobile app at the Node backend's public URL. Keep the Python service off the public internet — only Node's container network (or `localhost` if same machine) needs to reach it.

---

## 8. Notes & Next Steps

- **Model loading**: all three models load once at Python service startup (see `app/main.py`) — never reload them per-request, that's the most common performance mistake in serving setups like this.
- **Fertilizer input validation**: the Pydantic `FertilizerInput` schema means malformed requests get rejected with a clear error before reaching the model — Node currently just relays that error message through; worth mapping it to a friendlier message in the frontend.
- **SSP class caveat**: as noted when the model was trained, `SSP` is underrepresented in the training data and has the weakest recall — consider flagging low-confidence fertilizer predictions in the UI (e.g. if the top probability is below ~0.5, show the top-3 instead of a single answer).
- **Combining outputs**: since disease, pest, and fertilizer are separate endpoints, the app/web page decides how to present them together (e.g. a single "scan my field" flow that calls `/api/disease` and `/api/pest` on the same photo, plus a separate form for `/api/fertilizer`) — Node doesn't need to know about that orchestration, it just proxies each call.
- **Authentication**: this guide has no auth on the endpoints. If the app will have user accounts, that's exactly the kind of logic that belongs in Node (JWT/session check) before it forwards a request to the Python service — keep the Python service auth-free and trusting only Node.
- **File uploads on Node**: `multer` here uses in-memory storage (`memoryStorage()`), which is fine for typical phone-camera image sizes — for very large uploads or high traffic, switch to disk storage or a direct-to-storage upload pattern instead.
