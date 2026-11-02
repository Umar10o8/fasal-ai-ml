import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import FertilizerInput

MODEL_PATH = Path(__file__).parent / "models" / "fertilizer_rf_model.pkl"
NODE_BACKEND_URL = os.getenv("NODE_BACKEND_URL", "http://localhost:5000")

state = {"model": None, "categories": {}, "error": None}


def encoder_categories(model):
    """Allowed values per text column, read from the pipeline's OneHotEncoder."""
    categories = {}
    for _name, transformer, columns in model.named_steps["preprocessor"].transformers_:
        if hasattr(transformer, "categories_"):
            for column, values in zip(columns, transformer.categories_):
                categories[column] = [str(v) for v in values]
    return categories


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup. If it fails the service still starts, so /health can say why.
    try:
        model = joblib.load(MODEL_PATH)
        state["categories"] = encoder_categories(model)
        state["model"] = model
    except Exception as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
    yield
    state["model"] = None


app = FastAPI(title="Fasal.AI ML Service (internal)", version="0.1.0", lifespan=lifespan)

# Internal service: only the Node backend calls it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[NODE_BACKEND_URL],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    loaded = state["model"] is not None
    body = {"status": "ok" if loaded else "degraded", "fertilizer_model_loaded": loaded}
    if state["error"]:
        body["error"] = state["error"]
    return JSONResponse(status_code=200 if loaded else 503, content=body)


@app.post("/predict/fertilizer")
def predict_fertilizer(data: FertilizerInput):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Fertilizer model is not loaded")

    values = data.model_dump()

    # The encoder ignores unknown categories, which would silently skew the prediction.
    errors = [
        {"loc": ["body", column], "msg": f"must be one of: {', '.join(allowed)}"}
        for column, allowed in state["categories"].items()
        if values[column] not in allowed
    ]
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    # Same column order the pipeline was fitted with.
    input_df = pd.DataFrame([values])[list(model.feature_names_in_)]

    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    classes = model.named_steps["classifier"].classes_
    top3_idx = proba.argsort()[::-1][:3]
    top3 = [{"fertilizer": str(classes[i]), "probability": round(float(proba[i]), 3)} for i in top3_idx]

    return {"recommended_fertilizer": str(prediction), "top3": top3}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=True)
