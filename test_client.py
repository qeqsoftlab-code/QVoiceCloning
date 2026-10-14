import sys
import requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:8001"

def test_service():
    print(f"[1/3] A verificar estado em {BASE_URL}/health...")
    try:
        res = requests.get(f"{BASE_URL}/health", timeout=10)
        print("Status:", res.status_code, res.json())
    except Exception as e:
        print(f"Erro ao conectar com o serviço: {e}")
        return

    print("\n[2/3] A verificar vozes disponíveis em /voices...")
    res = requests.get(f"{BASE_URL}/voices")
    voices = res.json().get("available_voices", [])
    print("Vozes disponíveis:", voices)

    if not voices:
        print("Nenhuma voz .wav encontrada em /voices. Coloque 'default.wav' na pasta para testar síntese.")
        return

    voice = voices[0]
    print(f"\n[3/3] A testar síntese de voz usando a referência '{voice}'...")
    payload = {
        "text": "Este é um teste de clonagem de voz local executado através de contentor Docker com aceleração por GPU.",
        "voice_name": voice,
        "language": "pt"
    }
    synth_res = requests.post(f"{BASE_URL}/synthesize", json=payload, timeout=180)
    if synth_res.status_code == 200:
        out_path = Path("test_output.wav")
        out_path.write_bytes(synth_res.content)
        print(f"Sucesso! Áudio gerado e guardado em {out_path.resolve()} ({len(synth_res.content)} bytes)")
    else:
        print("Falha na síntese:", synth_res.status_code, synth_res.text)

if __name__ == "__main__":
    test_service()
