# QVoiceCloning Microservice

Microserviço autônomo de síntese e clonagem de voz baseado em **Coqui XTTS-v2**, empacotado para execução em container Docker com suporte a aceleração por GPU NVIDIA (CUDA).

---

## 1. Arquitetura e Características

- **Totalmente Isolado**: Independente de outras aplicações ou studios web.
- **Engine TTS**: Coqui XTTS-v2 com suporte a clonagem zero-shot a partir de amostras de áudio de referência (.wav).
- **Idioma Padrão**: Português (`pt`), extensível para mais de 16 idiomas suportados pelo XTTS-v2.
- **Padrão Singleton**: O modelo de IA é carregado em memória VRAM/RAM apenas uma vez no startup da API, evitando recargas a cada requisição.
- **Persistência de Cache**: O modelo baixado (`/root/.local/share/tts`) é persistido na pasta local `./models`, prevenindo downloads repetidos a cada reinicialização.
- **API REST de Alta Performance**: Desenvolvida com FastAPI e Uvicorn na porta `8001`.

---

## 2. Estrutura do Projeto

```text
QVoiceCloning/
├── core/
│   ├── __init__.py
│   └── synthesizer.py       # Gerenciador Singleton do modelo XTTS-v2
├── voices/                  # Amostras de voz de referência (.wav)
│   └── .gitkeep
├── output/                  # Áudios sintetizados gerados
│   └── .gitkeep
├── models/                  # Volume de cache persistente do modelo XTTS-v2
│   └── .gitkeep
├── server.py                # Servidor FastAPI com rotas /health, /voices, /synthesize
├── requirements.txt         # Dependências Python (PyTorch CUDA 11.8, Coqui TTS, etc.)
├── Dockerfile               # Imagem Ubuntu 22.04 com runtime CUDA 11.8 e FFmpeg
├── compose.yaml             # Configuração do container Docker e reserva de GPU
├── test_client.py           # Script para validação de endpoints e teste de síntese
└── README.md
```

---

## 3. Pré-requisitos

1. **Docker Desktop** (com backend WSL 2 ativo no Windows).
2. **Placa de Vídeo NVIDIA** com suporte a CUDA e drivers atualizados instalados no host.
3. **NVIDIA Container Toolkit** habilitado no Docker Desktop (já integrado por padrão nas versões recentes do Docker Desktop para Windows).

> **Nota sobre Fallback para CPU:**  
> Caso execute em um ambiente sem GPU NVIDIA dedicada, comente a seção `deploy.resources.reservations` em [compose.yaml](file:///c:/Users/thomas.quadros/Documents/Thomas/Projetos%20Pessoais/QVoiceCloning/compose.yaml). O código do sintetizador detecta automaticamente `torch.cuda.is_available()` e utiliza CPU como fallback.

---

## 4. Como Executar

### Passo 1: Preparar uma voz de referência
Coloque pelo menos um ficheiro de áudio `.wav` (idealmente de 6 a 15 segundos de voz limpa e sem ruídos de fundo) dentro da pasta `voices/`. Por exemplo:
```text
voices/default.wav
```

### Passo 2: Construir e iniciar o container
No terminal (PowerShell ou Bash), navegue até a pasta do projeto e execute:

```powershell
docker compose up --build -d
```

### Passo 3: Monitorar o download inicial e inicialização
No primeiro arranque, o Coqui TTS fará o download dos pesos do modelo XTTS-v2 (~1.8 GB) para a pasta `./models`. Acompanhe os logs até ver a mensagem de sucesso:

```powershell
docker compose logs -f qvoicecloning
```

Você verá a mensagem:
```text
[QVoiceCloning] A inicializar XTTS-v2 no dispositivo: cuda
[QVoiceCloning] Modelo carregado com sucesso em memória VRAM.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

---

## 5. Endpoints da API

A documentação interativa Swagger UI estará acessível em:
**http://localhost:8001/docs**

### `GET /health`
Verifica a integridade do microserviço e se a aceleração CUDA está ativa:
```json
{
  "status": "online",
  "service": "QVoiceCloning",
  "cuda_available": true,
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX ..."
}
```

### `GET /voices`
Lista as vozes disponíveis na pasta `./voices`:
```json
{
  "available_voices": ["default"]
}
```

### `POST /synthesize`
Sintetiza texto utilizando a voz informada e retorna o ficheiro de áudio `.wav`.

**Corpo da Requisição (JSON):**
```json
{
  "text": "Olá! Esta é uma demonstração de síntese de voz usando o microserviço QVoiceCloning.",
  "voice_name": "default",
  "language": "pt",
  "temperature": 0.82,
  "repetition_penalty": 2.0,
  "top_p": 0.85,
  "length_penalty": 1.0
}
```
**Resposta:** Arquivo binário de áudio `audio/wav`.

---

## 6. Teste Automatizado com `test_client.py`

Com o container em execução, execute o script de teste:

```powershell
python test_client.py
```

O script irá:
1. Validar o endpoint `/health`;
2. Listar as vozes em `/voices`;
3. Disparar uma síntese com o primeiro arquivo encontrado e salvar localmente como `test_output.wav`.

---

## 7. Comandos Úteis do Docker

- **Parar o container:**
  ```powershell
  docker compose down
  ```
- **Reiniciar o serviço:**
  ```powershell
  docker compose restart
  ```
- **Verificar status dos containers:**
  ```powershell
  docker compose ps
  ```
- **Verificar uso de GPU no container:**
  ```powershell
  docker compose exec qvoicecloning nvidia-smi
  ```
