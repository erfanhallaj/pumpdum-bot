# PowerShell script to run bot as daemon (always running)
# Restarts automatically on crash

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AI Pump/Dump Bot - Daemon Mode" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$maxRestarts = 1000
$restartCount = 0
$restartDelay = 10

while ($restartCount -lt $maxRestarts) {
    try {
        Write-Host "🔄 Starting bot (attempt $($restartCount + 1))..." -ForegroundColor Yellow
        Write-Host "----------------------------------------" -ForegroundColor Gray
        
        # Run the bot
        $process = Start-Process -FilePath "python" -ArgumentList "main.py" -NoNewWindow -PassThru -Wait
        
        if ($process.ExitCode -eq 0) {
            Write-Host "✅ Bot stopped normally" -ForegroundColor Green
            break
        } else {
            $restartCount++
            Write-Host "⚠️  Bot crashed (exit code: $($process.ExitCode))" -ForegroundColor Red
            Write-Host "🔄 Restarting in $restartDelay seconds... (restart $restartCount/$maxRestarts)" -ForegroundColor Yellow
            Start-Sleep -Seconds $restartDelay
        }
    } catch {
        $restartCount++
        Write-Host "❌ Error: $_" -ForegroundColor Red
        Write-Host "🔄 Restarting in $restartDelay seconds... (restart $restartCount/$maxRestarts)" -ForegroundColor Yellow
        Start-Sleep -Seconds $restartDelay
    }
}

if ($restartCount -ge $maxRestarts) {
    Write-Host "❌ Maximum restart attempts reached. Stopping." -ForegroundColor Red
}

Read-Host "Press Enter to exit"

