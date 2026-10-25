<#
.SYNOPSIS
    Keeps one shared FusionVisionMCP server running on http://127.0.0.1:8765/mcp, restarting it if it exits.

.DESCRIPTION
    For Windows, where Task Scheduler starts programs but does not restart one that exits. Run it at logon
    (see docs/shared_server.md). Each start is appended to the log with its exit code, and restarts back
    off from 5 s to 5 min so a server that cannot start does not spin.

.EXAMPLE
    powershell -NoProfile -File scripts\run_shared_server.ps1 -- --memory-mode standard
#>
param(
    [string]$Exe = "$env:USERPROFILE\.local\bin\fusion-vision-mcp.exe",
    [int]$Port = 8765,
    [string]$Log = "$env:LOCALAPPDATA\FusionVisionMCP\shared_server.log",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ServerArgs = @()
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force (Split-Path $Log) | Out-Null
$delay = 5
while ($true) {
    $started = Get-Date
    Add-Content $Log "$($started.ToString('s')) starting $Exe on port $Port $ServerArgs"
    $proc = Start-Process -FilePath $Exe -NoNewWindow -PassThru -Wait `
        -ArgumentList (@("--transport", "streamable-http", "--port", "$Port") + $ServerArgs) `
        -RedirectStandardError "$Log.stderr"
    Get-Content "$Log.stderr" -ErrorAction SilentlyContinue | Add-Content $Log
    Add-Content $Log "$((Get-Date).ToString('s')) exited with code $($proc.ExitCode)"
    # A run that lasted a while was a real server, so start over from the short delay.
    if (((Get-Date) - $started).TotalMinutes -gt 10) { $delay = 5 }
    Start-Sleep -Seconds $delay
    $delay = [Math]::Min($delay * 2, 300)
}
