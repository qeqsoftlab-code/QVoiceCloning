import os
import re
import torch
from TTS.api import TTS

class QVoiceSynthesizer:
    _instance = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        print(f"\n[QVoiceCloning] A inicializar XTTS-v2 no dispositivo: {self.device}")
        
        # Carrega o modelo com aceitação explícita dos termos de uso da Coqui
        os.environ["COQUI_TOS_AGREED"] = "1"
        self.tts = TTS(model_name=self.model_name, progress_bar=False).to(self.device)
        print("[QVoiceCloning] Modelo carregado com sucesso em memória VRAM.\n")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = QVoiceSynthesizer()
        return cls._instance

    @staticmethod
    def _split_into_chunks(text: str) -> list:
        # Se contiver pontuações terminais padrão (. ! ?)
        if re.search(r'[.!?]', text):
            raw = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]
        else:
            # Caso os pontos tenham sido convertidos em vírgulas antes de letras maiúsculas
            raw = [s.strip() for s in re.split(r'(?:,\s+|\n+)(?=[A-ZÀ-ÿ])', text) if s.strip()]
        
        chunks = []
        for r in raw:
            # Remove qualquer ponto terminal residual para prevenir o bug de pronúncia do XTTS
            r = re.sub(r'\.+', '', r).strip()
            # Se algum chunk ainda exceder 220 caracteres, divide suavemente por vírgula
            if len(r) > 220 and ',' in r:
                sub = [s.strip() for s in r.split(',') if s.strip()]
                chunks.extend(sub)
            elif r:
                chunks.append(r)
        return chunks if chunks else [text]

    def synthesize(
        self,
        text: str,
        speaker_wav: str,
        output_path: str,
        language: str = "pt",
        temperature: float = 0.82,
        repetition_penalty: float = 2.0,
        top_p: float = 0.85,
        length_penalty: float = 1.0,
    ) -> str:
        if not os.path.exists(speaker_wav):
            raise FileNotFoundError(f"Ficheiro de referência não encontrado: {speaker_wav}")

        chunks = self._split_into_chunks(text)
        print(f"[QVoiceCloning] A sintetizar {len(chunks)} sentenças/trechos...")

        all_wavs = []
        sample_rate = 24000
        # Pausa sintática natural de 0.20s entre sentenças (4800 amostras a 24kHz)
        pause = [0.0] * int(sample_rate * 0.20)

        for i, chunk in enumerate(chunks):
            wav_chunk = self.tts.tts(
                text=chunk,
                speaker_wav=speaker_wav,
                language=language,
                split_sentences=False,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                length_penalty=length_penalty,
            )
            all_wavs.extend(wav_chunk)
            if i < len(chunks) - 1:
                all_wavs.extend(pause)

        self.tts.synthesizer.save_wav(wav=all_wavs, path=output_path)
        return output_path
