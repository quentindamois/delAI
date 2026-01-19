#!/usr/bin/env pwsh

# Active le venv
& "$PSScriptRoot\venv\Scripts\Activate.ps1"

# Charge les variables d'environnement depuis discord.env
Get-Content "$PSScriptRoot\students-assistant\discord.env" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}

# Force Discord uniquement et définis l'endpoint Flask local
$env:ENABLE_DISCORD_BOT = "1"
$env:ENABLE_TEAMS_BOT = "0"
$env:LLM_ENDPOINT = "http://localhost:5000/ask"

# Lance le bot
Set-Location "$PSScriptRoot\students-assistant"
python src/main.py
