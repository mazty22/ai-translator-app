import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai

# ---- Setup ----
app = FastAPI()

# Allow the frontend (served from anywhere) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reads your API key from the GEMINI_API_KEY environment variable automatically
# (never hardcode it in code!)
client = genai.Client()


class TranslateRequest(BaseModel):
    text: str
    target_language: str


@app.post("/translate")
def translate(req: TranslateRequest):
    prompt = (
        f"Translate the following text into {req.target_language}. "
        f"Only return the translated text, nothing else.\n\n"
        f"Text: {req.text}"
    )

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    translated_text = response.text
    return {"translation": translated_text}


# ---- Serve the frontend (index.html) at the root URL ----
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
