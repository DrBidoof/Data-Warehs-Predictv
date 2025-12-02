# Data-Warehs-Predictv — Frontend + Backend

This repo contains a small Express backend and a Vite+React frontend that provides three tabs to run the Python scripts in the repository root:

- `Data exploration.py`
- `Data modelling.py`
- `Predictive model building.py`

The backend executes these scripts and returns stdout/stderr so the frontend can display results.

Prerequisites (Windows PowerShell):
- Node.js (recommend v18+)
- Python installed and available as `python` in PATH

Quick start (PowerShell):

1) Install backend deps and start backend (Flask)

```powershell
cd "C:/Users/endya/Desktop/4th Sem/COMP309-W/Data-Warehs-Predictv"
# (optional) create and activate venv
# python -m venv .venv
# .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python backend/app.py
```

Backend will listen on `http://localhost:5000`.

2) Install frontend deps and start frontend dev server

```powershell
cd ../frontend
npm install
npm run dev
```

Open the displayed Vite URL (default `http://localhost:5173`) and click the tabs to run scripts.

Notes:
- The backend runs the Python scripts from the repo root. Ensure the script files are present and that Python is in your PATH.
- If your Python command is `py` instead of `python`, edit `backend/server.js` and replace the `python` command accordingly.

If you want, I can also add a small wrapper to stream progressive output, run tests, or create nicer UI/UX for long-running jobs.
