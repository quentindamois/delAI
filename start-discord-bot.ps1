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

# Charge les variables d'environnement depuis discord.env
Get-Content "$PSScriptRoot\flask_chatbot_student_assistant\.env" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}

# Force Discord uniquement et définis l'endpoint Flask local
$env:ENABLE_DISCORD_BOT = "1"
$env:ENABLE_TEAMS_BOT = "0"
$env:LLM_ENDPOINT = "http://localhost:5000/ask"

# Lance le email receiver en arrière-plan
Write-Host "Starting email receiver..."
Set-Location "$PSScriptRoot\flask_chatbot_student_assistant"
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "email_receiver.py"

# Lance le bot
Set-Location "$PSScriptRoot\students-assistant"
python src/main.py
