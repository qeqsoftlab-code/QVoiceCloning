import os
import re
from pathlib import Path
import numpy as np
import torch
from TTS.api import TTS

class QVoiceSynthesizer:
    _instance = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        self._latents_cache = {}
        
        # Otimização de threads no CPU
        if self.device == "cpu":
            num_cores = os.cpu_count() or 4
            torch.set_num_threads(num_cores)
            print(f"[QVoiceCloning] Configurado torch.set_num_threads({num_cores})")
            
        print(f"\n[QVoiceCloning] A inicializar XTTS-v2 no dispositivo: {self.device}")
        
        # Carrega o modelo com aceitação explícita dos termos de uso da Coqui
        os.environ["COQUI_TOS_AGREED"] = "1"
        self.tts = TTS(model_name=self.model_name, progress_bar=False).to(self.device)
        print("[QVoiceCloning] Modelo carregado com sucesso em memória.\n")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = QVoiceSynthesizer()
        return cls._instance

    def get_speaker_latents(self, speaker_wav: str):
        wav_path = Path(speaker_wav)
        latents_path = wav_path.with_suffix(".latents.pth")
        
        # 1. Cache em memória RAM
        if speaker_wav in self._latents_cache:
            return self._latents_cache[speaker_wav]
            
        # 2. Cache persistente em arquivo .pth
        if latents_path.exists():
            try:
                if wav_path.stat().st_mtime <= latents_path.stat().st_mtime:
                    data = torch.load(str(latents_path), map_location=self.device)
                    gpt_cond_latent = data["gpt_cond_latent"]
                    speaker_embedding = data["speaker_embedding"]
                    self._latents_cache[speaker_wav] = (gpt_cond_latent, speaker_embedding)
                    print(f"[QVoiceCloning] Latentes do locutor carregados do cache: {latents_path.name}")
                    return gpt_cond_latent, speaker_embedding
            except Exception as e:
                print(f"[QVoiceCloning] Aviso: erro ao ler cache {latents_path.name}: {e}")
                
        # 3. Cálculo de latentes (primeira vez)
        print(f"[QVoiceCloning] A calcular latentes de locutor para: {wav_path.name}...")
        gpt_cond_latent, speaker_embedding = self.tts.synthesizer.tts_model.get_conditioning_latents(
            audio_path=[str(wav_path)],
            max_ref_length=30,
            gpt_cond_len=6,
            gpt_cond_chunk_len=6,
        )
        self._latents_cache[speaker_wav] = (gpt_cond_latent, speaker_embedding)
        
        # Salva em disco para persistir entre reinicializações
        try:
            torch.save(
                {"gpt_cond_latent": gpt_cond_latent, "speaker_embedding": speaker_embedding},
                str(latents_path)
            )
            print(f"[QVoiceCloning] Latentes persistidos em disco: {latents_path.name}")
        except Exception as e:
            print(f"[QVoiceCloning] Aviso: não foi possível gravar {latents_path.name}: {e}")
            
        return gpt_cond_latent, speaker_embedding

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
            # Se algum chunk exceder 120 caracteres, divide suavemente por vírgula para fluidez
            if len(r) > 120 and ',' in r:
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
        temperature: float = 0.75,
        repetition_penalty: float = 5.0,
        top_p: float = 0.85,
        top_k: int = 30,
        speed: float = 1.0,
        length_penalty: float = 1.0,
    ) -> str:
        if not os.path.exists(speaker_wav):
            raise FileNotFoundError(f"Ficheiro de referência não encontrado: {speaker_wav}")

        chunks = self._split_into_chunks(text)
        print(f"[QVoiceCloning] A sintetizar {len(chunks)} sentenças/trechos...")

        gpt_cond_latent, speaker_embedding = self.get_speaker_latents(speaker_wav)

        all_wavs = []
        sample_rate = 24000
        # Pausa sintática natural de 0.20s entre sentenças (4800 amostras a 24kHz)
        pause = [0.0] * int(sample_rate * 0.20)

        with torch.inference_mode():
            for i, chunk in enumerate(chunks):
                print(f"[QVoiceCloning] Trecho {i+1}/{len(chunks)}: '{chunk}'")
                out = self.tts.synthesizer.tts_model.inference(
                    text=chunk,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=temperature,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                    top_k=top_k,
                    speed=speed,
                    enable_text_splitting=False,
                )
                wav_chunk = out["wav"]
                if torch.is_tensor(wav_chunk):
                    wav_chunk = wav_chunk.cpu().numpy()
                if hasattr(wav_chunk, "tolist"):
                    wav_chunk = wav_chunk.tolist()
                elif isinstance(wav_chunk, np.ndarray):
                    wav_chunk = list(wav_chunk)

                all_wavs.extend(wav_chunk)
                if i < len(chunks) - 1:
                    all_wavs.extend(pause)

        self.tts.synthesizer.save_wav(wav=all_wavs, path=output_path)
        return output_path
