#!/usr/bin/env pwsh

# Active le venv
& "$PSScriptRoot\venv\Scripts\Activate.ps1"

# Force unbuffered output
$env:PYTHONUNBUFFERED = "1"

# Lance Flask
Set-Location "$PSScriptRoot\flask_chatbot_student_assistant"
python app.py
