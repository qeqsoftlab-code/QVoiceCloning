#!/bin/bash
PID=$(lsof -ti :8001)
if [ -n "$PID" ]; then
    kill -9 $PID
    echo "QVoiceCloning (PID $PID) encerrado com sucesso."
else
    echo "Nenhum processo rodando na porta 8001."
fi
