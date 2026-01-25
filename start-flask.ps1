#!/usr/bin/env pwsh

# Active le venv
if (Test-Path "$PSScriptRoot\.venv\Scripts\Activate.ps1") {
    & "$PSScriptRoot\.venv\Scripts\Activate.ps1"
} elseif (Test-Path "$PSScriptRoot\venv\Scripts\Activate.ps1") {
    & "$PSScriptRoot\venv\Scripts\Activate.ps1"
} else {
    Write-Error "Aucun venv trouvé (.venv ou venv)"
}

# Force unbuffered output
$env:PYTHONUNBUFFERED = "1"

# Lance Flask
Set-Location "$PSScriptRoot\flask_chatbot_student_assistant"
python app.py
