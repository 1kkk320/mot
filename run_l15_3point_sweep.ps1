$ErrorActionPreference = 'Stop'

$presets = @('weaker', 'current', 'stronger')

foreach ($preset in $presets) {
    Write-Host "=== Running L1.5 risk preset: $preset ==="
    $env:L15_RISK_PRESET = $preset
    $outputPath = Join-Path 'logs' ("l15_3point_" + $preset + ".txt")
    conda run -n mot python main.py 2>&1 | Tee-Object -FilePath $outputPath
}

Remove-Item Env:L15_RISK_PRESET -ErrorAction SilentlyContinue
