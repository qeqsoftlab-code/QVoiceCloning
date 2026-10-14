import io
import re
import sys
import time
import requests
from pathlib import Path

# Configuração de encoding para evitar erros de charmap no Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_URL = "http://127.0.0.1:8001"

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

sentences = [
    "Não vejo necessidade disso.",
    "Todos somos livres, adultos e responsáveis pelo que publicamos, e sabemos que, ao expor uma opinião, estamos sujeitos a questionamentos e discordâncias.",
    "Para mim, amizade também é isso: poder discordar sem que isso se torne um problema.",
    "Se alguém quer apenas pessoas que concordem com tudo o que diz, não está procurando amigos, mas seguidores.",
    "Se publico algo, é perfeitamente natural que algumas pessoas concordem e outras questionem com base em suas próprias visões de mundo.",
    "E, se publico algo em um grupo onde sei que há opiniões amplamente contrárias às minhas, esperar uma resposta imediata faz parte da brincadeira.",
    "Somos adultos, afinal.",
    "Discordar, debater e até provocar um pouco faz parte da convivência, desde que haja respeito e maturidade para lidar com as consequências do que se diz."
]

raw_text = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
clean_text = sanitize_tts_input(raw_text)
voice_name = "landscape_estudio"

def main():
    print(f"[*] Texto original ({len(raw_text)} caracteres):")
    print(f"    {raw_text[:120]}...\n")
    print(f"[*] Texto sanitizado ({len(clean_text)} caracteres):")
    print(f"    {clean_text[:120]}...\n")
    print(f"[*] Disparando síntese com a voz '{voice_name}'...")
    print("[*] Aguardando processamento da API...")

    start_time = time.time()
    try:
        payload = {
            "text": clean_text,
            "voice_name": voice_name,
            "language": "pt",
            "temperature": 0.82,
            "repetition_penalty": 2.0,
            "top_p": 0.85,
            "length_penalty": 1.0
        }
        response = requests.post(
            f"{BASE_URL}/synthesize",
            json=payload,
            timeout=1800  # 30 minutos de margem para inferência em CPU
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            out_file = Path("output_landscape_estudio_expressive.wav")
            out_file.write_bytes(response.content)
            Path("output_landscape_estudio.wav").write_bytes(response.content)
            print(f"\n[+] Sucesso! Áudio gerado em {elapsed:.1f}s.")
            print(f"[+] Arquivo salvo em: {out_file.resolve()}")
            print(f"[+] Tamanho: {len(response.content)} bytes.")
        else:
            print(f"\n[-] Erro na resposta ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"\n[-] Erro ao enviar requisição: {e}")

if __name__ == "__main__":
    main()
