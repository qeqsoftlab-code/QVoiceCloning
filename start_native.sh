#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if lsof -i :8001 >/dev/null 2>&1; then
    echo "QVoiceCloning já está ativo na porta 8001."
    exit 0
fi

echo "Iniciando QVoiceCloning nativo no Apple Silicon (porta 8001)..."
nohup .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8001 > uvicorn.log 2>&1 &
echo "Servidor iniciado em segundo plano (PID $!). Acompanhe os logs com: tail -f uvicorn.log"
