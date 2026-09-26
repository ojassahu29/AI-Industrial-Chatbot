# AI-Powered Industrial Chatbot

Course mini-project: **AI-Powered Industrial Chatbot Using Hugging Face Large Language Models**

## Architecture

Documents → text extraction → chunking → sentence embeddings → cosine-similarity retrieval → prompt engineering → Qwen2.5-1.5B-Instruct → CLI/Streamlit UI.

## Models

- LLM: `Qwen/Qwen2.5-1.5B-Instruct`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`

## Project structure

```text
AI_Industrial_Chatbot_Project/
├── app.py
├── requirements.txt
├── README.md
├── REPORT.pdf
├── data/
│   ├── industrial_robot_safety.txt
│   ├── cnc_machine_operation.txt
│   └── plc_maintenance.txt
├── src/
│   ├── config.py
│   ├── rag.py
│   ├── build_index.py
│   └── cli.py
├── vector_store/
└── screenshots/
```

## Windows setup

```powershell
cd AI_Industrial_Chatbot_Project

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## Build the knowledge index

```powershell
python src/build_index.py
```

## Run CLI

```powershell
python src/cli.py
```

## Run web UI

```powershell
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## Suggested demonstration questions

1. What precautions are required before entering a robot workcell?
2. What should be checked before restarting the CNC machine?
3. What are the symptoms of PLC communication failure?
4. What should the operator do if the requested information is not in the documents?

The last question demonstrates the grounding/fallback behavior.

## Important

The sample documents in `data/` are project demonstration documents because the assignment brief supplied for this project does not include a separate technical document set. Replace them with the instructor-provided documents if such documents are supplied.
