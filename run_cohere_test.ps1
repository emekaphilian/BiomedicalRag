# Set Cohere API key and run test
$env:COHERE_API_KEY="your-cohere-api-key-here"

Write-Host "Running Cohere integration test with fixed imports..." -ForegroundColor Cyan

.\venv\Scripts\python.exe stabilization_test_cohere.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Cohere integration test passed!" -ForegroundColor Green
} else {
    Write-Host "`n[FAILED] Test exit code: $LASTEXITCODE" -ForegroundColor Red
}
