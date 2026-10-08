# FraudShield AI V4 — clickable navigation

This version replaces the decorative sidebar navigation and duplicate main-panel tabs with **one working Streamlit sidebar radio navigation**. Clicking Overview, Investigation, Model Lab, Simulator, or Monitoring switches the main content panel. All V3 capabilities, trained model, evaluation results, and API are retained.

## Windows PowerShell

Extract the ZIP, then `cd .\FraudShield_AI_V4` from the outer folder. If your virtual environment is in the outer folder:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Or create a fresh environment inside the V4 project:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Open http://localhost:8501. Stop the previous dashboard with Ctrl+C first. The model artifact is included; the raw ULB credit-card dataset is excluded.

This is a research demonstration, not a production banking system.
