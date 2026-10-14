import os
import re
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from core.synthesizer import QVoiceSynthesizer

app = FastAPI(title="QVoiceCloning API", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
VOICES_DIR = BASE_DIR / "voices"
OUTPUT_DIR = BASE_DIR / "output"

VOICES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def sanitize_tts_input(text: str) -> str:
    cleaned = text.replace('\r\n', ' ').replace('\n', ' ')
    cleaned = cleaned.replace(':', ',')
    # Remove qualquer injeção acidental da palavra literal
    cleaned = re.sub(r'\b(ponto final|ponto)\b\.?', '', cleaned, flags=re.IGNORECASE)
    
    # Solução para o bug do tokenizador pt do XTTS:
    # Substitui pontos finais declarativos por vírgula e espaço ou reticências leves
    # para gerar a pausa sintática exata sem engatilhar a leitura do vocábulo "ponto"
    cleaned = re.sub(r'\s*\.\s*', ', ', cleaned)
    cleaned = re.sub(r'\s*,\s*', ', ', cleaned)
    cleaned = re.sub(r',\s*,+', ', ', cleaned)
    cleaned = re.sub(r'\s+([,!?;])', r'\1', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'[,\s]+$', '', cleaned)
    return cleaned.strip()

from typing import Optional

class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Texto a sintetizar")
    voice_name: str = Field(default="default", description="Nome do ficheiro em /voices sem a extensão .wav")
    language: str = Field(default="pt", description="Código do idioma (pt para Português)")
    temperature: Optional[float] = Field(default=0.82, description="Modulação de pitch e expressividade")
    repetition_penalty: Optional[float] = Field(default=2.0, description="Penalidade de repetição para evitar monotonia rítmica")
    top_p: Optional[float] = Field(default=0.85, description="Filtro top_p de amostragem de tokens")
    length_penalty: Optional[float] = Field(default=1.0, description="Penalidade de extensão")

@app.on_event("startup")
def startup_event():
    # Inicializa o modelo no arranque da API
    QVoiceSynthesizer.get_instance()

@app.get("/health")
def health_check():
    import torch
    return {
        "status": "online",
        "service": "QVoiceCloning",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }

@app.get("/voices")
def list_voices():
    voices = [f.stem for f in VOICES_DIR.glob("*.wav")]
    return {"available_voices": voices}

@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    voice_path = VOICES_DIR / f"{req.voice_name}.wav"
    if not voice_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Ficheiro de áudio '{req.voice_name}.wav' não encontrado na pasta voices/."
        )

    out_file = OUTPUT_DIR / f"voice_{uuid.uuid4().hex[:8]}.wav"
    try:
        sanitized_text = sanitize_tts_input(req.text)
        engine = QVoiceSynthesizer.get_instance()
        engine.synthesize(
            text=sanitized_text,
            speaker_wav=str(voice_path),
            output_path=str(out_file),
            language=req.language,
            temperature=req.temperature if req.temperature is not None else 0.82,
            repetition_penalty=req.repetition_penalty if req.repetition_penalty is not None else 2.0,
            top_p=req.top_p if req.top_p is not None else 0.85,
            length_penalty=req.length_penalty if req.length_penalty is not None else 1.0,
        )
        return FileResponse(
            path=str(out_file),
            media_type="audio/wav",
            filename=out_file.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na síntese de voz: {str(e)}")
