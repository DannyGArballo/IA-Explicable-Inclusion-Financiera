$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m streamlit run .\streamlit_app.py
