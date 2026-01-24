# Active le venv
& "$PSScriptRoot\venv\Scripts\Activate.ps1"

# Force unbuffered output
$env:PYTHONUNBUFFERED = "1"

# Charge les variables d'environnement depuis discord.env
Get-Content "$PSScriptRoot\streamlit_interface\cmd.env" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}

# Lance le bot
Set-Location "$PSScriptRoot\streamlit_interface"
streamlit run main.py