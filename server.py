import os
import re
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
from core.synthesizer import QVoiceSynthesizer

app = FastAPI(title="QVoiceCloning API", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
VOICES_DIR = BASE_DIR / "voices"
OUTPUT_DIR = BASE_DIR / "output"

VOICES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

def sanitize_tts_input(text: str) -> str:
    if not text:
        return ""
    # 1. Expansão de siglas literárias e conhecidas
    text = re.sub(r"J\.?\s*R\.?\s*R\.?", "Jota Érre Érre", text, flags=re.IGNORECASE)
    text = re.sub(r"([A-Z])\.(?=[A-Z]\.?)", r" ", text)
    
    # 2. Higienização fonética fina: evita tensão no 's' final em palavras terminadas em s/as/os antes de pontuação
    text = re.sub(r"([a-zA-ZÀ-ÿ]+[sS])\s*[.,;!?:-]+\s*", r" ", text)
    # Demais pontuações normais com ponto final
    text = re.sub(r"([a-zA-ZÀ-ÿ])\.\s*", r", ", text)
    
    # 3. Limpeza de resíduos de pontuação e espaços
    text = text.replace(":", ",")
    text = re.sub(r"(ponto final|ponto)\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r"\s+([,!?;])", r"", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


from typing import Optional

class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Texto a sintetizar")
    voice_name: str = Field(default="default", description="Nome do ficheiro em /voices sem a extensão .wav")
    language: str = Field(default="pt", description="Código do idioma (pt para Português)")
    temperature: Optional[float] = Field(default=0.75, description="Modulação de pitch e expressividade")
    repetition_penalty: Optional[float] = Field(default=1.5, description="Penalidade de repetição (1.5 preserva harmônicos naturais)")
    top_p: Optional[float] = Field(default=0.85, description="Filtro top_p de amostragem de tokens")
    top_k: Optional[int] = Field(default=50, description="Filtro top_k de amostragem acústica (50 para riqueza tímbrica natural)")
    speed: Optional[float] = Field(default=1.0, description="Velocidade da fala (ex: 1.0 ou 1.05 para ritmo mais ágil)")
    length_penalty: Optional[float] = Field(default=1.0, description="Penalidade de extensão")

@app.on_event("startup")
def startup_event():
    # Inicializa o modelo no arranque da API
    engine = QVoiceSynthesizer.get_instance()
    # Pré-carrega e persiste os latentes de todas as vozes da pasta voices/
    for wav_file in VOICES_DIR.glob("*.wav"):
        try:
            print(f"[QVoiceCloning] A pré-aquecer latentes da voz '{wav_file.stem}'...")
            engine.get_speaker_latents(str(wav_file))
        except Exception as e:
            print(f"[QVoiceCloning] Aviso: erro ao pré-aquecer {wav_file.name}: {e}")

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
            temperature=req.temperature if req.temperature is not None else 0.75,
            repetition_penalty=req.repetition_penalty if req.repetition_penalty is not None else 1.5,
            top_p=req.top_p if req.top_p is not None else 0.85,
            top_k=req.top_k if req.top_k is not None else 50,
            speed=req.speed if req.speed is not None else 1.0,
            length_penalty=req.length_penalty if req.length_penalty is not None else 1.0,
        )
        return FileResponse(
            path=str(out_file),
            media_type="audio/wav",
            filename=out_file.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na síntese de voz: {str(e)}")
