# PY RAG

## Installation

First, clone the repository:
```bash
git clone https://github.com/kelvinmenegasse/py-rag.git
cd py-rag
```

Then, install the requirements:
For AI: Install ollama and download the model:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
```bash
ollama pull deepseek-r1:8b
```

For Python: Install venv, create a virtual environment, and install the requirements:
```bash
sudo apt install python3.12-venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Activate venv:
```bash
source venv/bin/activate
```
Run ollama:
```bash
ollama serve
```

In another terminal, run streamlit:
```bash
streamlit run app.py
```