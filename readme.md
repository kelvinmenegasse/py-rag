IA:
install ollama: curl -fsSL https://ollama.com/install.sh | sh
download and run model: ollama run deepseek-r1:8b

Python:
install venv: sudo apt install python3.12-venv
create venv (optional): python3 -m venv venv
activate venv (optional): source venv/bin/activate
install requirements: pip install -r requirements.txt

Execute script: streamlite run app.py