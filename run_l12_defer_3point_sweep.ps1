$ErrorActionPreference = 'Stop'

$presets = @('current', 'mild', 'moderate')

foreach ($preset in $presets) {
    Write-Host "=== Running L1 defer preset: $preset ==="
    $env:L12_DEFER_PRESET = $preset
    $mainOutputPath = Join-Path 'logs' ("l12_defer_" + $preset + "_main.txt")
    $evalOutputPath = Join-Path 'logs' ("l12_defer_" + $preset + "_eval.txt")

    conda run -n mot python -u main.py 2>&1 | Tee-Object -FilePath $mainOutputPath
    conda run -n mot python -u evaluate_mota_idswitch.py 2>&1 | Tee-Object -FilePath $evalOutputPath

    Copy-Item results\virconv_OCM\car_summary.txt (Join-Path 'logs' ("car_summary_l12_defer_" + $preset + ".txt")) -Force
    Copy-Item results\virconv_OCM\car_detailed.csv (Join-Path 'logs' ("car_detailed_l12_defer_" + $preset + ".csv")) -Force
    Copy-Item logs\l12_defer_diag.log (Join-Path 'logs' ("l12_defer_diag_" + $preset + ".log")) -Force
}

Remove-Item Env:L12_DEFER_PRESET -ErrorAction SilentlyContinue
