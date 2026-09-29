import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai

# ---- Setup ----
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reads the API key from the GEMINI_API_KEY environment variable automatically
client = genai.Client()


class TranslateRequest(BaseModel):
    text: str
    target_language: str
    source_language: str = "Auto-detect"


class ExchangeRequest(BaseModel):
    amount: float
    from_currency: str
    to_currency: str


@app.post("/translate")
def translate(req: TranslateRequest):
    text = req.text.strip()
    if not text:
        return {"translation": "", "detected_language": None}

    if req.source_language == "Auto-detect":
        prompt = (
            f"Detect the language of the following text, then translate it into {req.target_language}. "
            f"Respond in exactly this format, with no extra commentary:\n"
            f"DETECTED: <detected language name>\n"
            f"TRANSLATION: <translated text>\n\n"
            f"Text: {text}"
        )
    else:
        prompt = (
            f"Translate the following text from {req.source_language} into {req.target_language}. "
            f"Respond in exactly this format, with no extra commentary:\n"
            f"DETECTED: {req.source_language}\n"
            f"TRANSLATION: <translated text>\n\n"
            f"Text: {text}"
        )

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    raw = response.text.strip()
    detected = None
    translation = raw

    if "DETECTED:" in raw and "TRANSLATION:" in raw:
        try:
            detected_part, translation_part = raw.split("TRANSLATION:", 1)
            detected = detected_part.replace("DETECTED:", "").strip()
            translation = translation_part.strip()
        except ValueError:
            pass

    return {"translation": translation, "detected_language": detected}


@app.post("/exchange")
def exchange(req: ExchangeRequest):
    """
    Live currency conversion using the free Frankfurter API
    (no API key required, backed by European Central Bank rates).
    """
    try:
        url = "https://api.frankfurter.app/latest"
        params = {
            "amount": req.amount,
            "from": req.from_currency,
            "to": req.to_currency,
        }
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()

        converted = data["rates"].get(req.to_currency)
        rate = converted / req.amount if req.amount else None

        return {
            "converted_amount": converted,
            "rate": rate,
            "date": data.get("date"),
            "error": None,
        }
    except Exception as e:
        return {
            "converted_amount": None,
            "rate": None,
            "date": None,
            "error": f"Could not fetch exchange rate: {str(e)}",
        }


# ---- Serve the frontend ----
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
