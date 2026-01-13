# Projeto LASCON X

Projeto para a LASCON X, conectando a AI com a biblioteca do NEST com MCP

## Início rápido

### Pré-requisitos

* Python 3.8+
* Ollama instalado e em execução
* O modelo **qwen3:4b-instruct** (ou altere o código para o modelo de sua preferência em `mcp_client.py`)

### Instalação

**1) Clonar o repositório**

```bash
git clone https://github.com/arthurpbuenoUSP/projeto-lascon.git
cd projeto-lascon
```

**2) Instalar dependências**

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

### Rodar o exemplo

**1) Iniciar o servidor MCP**

```bash
./venv/bin/python mcp_server.py
```

**2) Executar o cliente MCP**

```bash
./venv/bin/python mcp_client.py
```


