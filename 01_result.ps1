param(
    [Parameter(Mandatory=$false)]
    [string]$Id
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$token = (Get-Content "$ScriptDir\config\api_token.txt" -Raw).Trim()
$headers = @{ Authorization = "Bearer $token" }

if (-not $Id) {
    $Id = Read-Host "Enter job ID"
}

$outFile = "$env:TEMP\comfy_result_$Id.webp"

try {
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/result/$Id" `
        -Headers $headers -OutFile $outFile -ErrorAction Stop
    Write-Host "Saved: $outFile"
    Start-Process $outFile
}
catch {
    if ($_.Exception.Response.StatusCode -eq 400) {
        Write-Host "Still processing"
    }
    elseif ($_.Exception.Response.StatusCode -eq 404) {
        Write-Host "Task not found"
    }
    elseif ($_.Exception.Response.StatusCode -eq 500) {
        Write-Host "Generation failed"
    }
    else {
        Write-Host "Error: $_"
    }
}
