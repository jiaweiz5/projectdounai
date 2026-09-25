from fastapi import FastAPI
from pydantic import BaseModel

from detector import predict_ai_involvement
from ad_detector import predict_covert_ad

app = FastAPI(title="XHS Verifier")


# ============================================================
# REQUEST FORMAT
# ============================================================

class AnalyzeRequest(BaseModel):
    text: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "0.1.0"
    }


# ============================================================
# ANALYZE
# ============================================================

@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    #layer 1
    authorship_result = predict_ai_involvement(
        request.text
    )

    #layer 2
    covert_ad_result = predict_covert_ad(
    request.text
)
    return {
        "authorship": authorship_result,
         "covert_ad": covert_ad_result
    }