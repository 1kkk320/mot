$ErrorActionPreference = 'Stop'

$presets = @('pos_heavier', 'balanced', 'motion_heavier', 'motion_heaviest')

foreach ($preset in $presets) {
    Write-Host "=== Running L1.5 motion-position weight preset: $preset ==="
    $env:L15_WEIGHT_PRESET = $preset
    $mainOutputPath = Join-Path 'logs' ("l15_weight_4point_" + $preset + "_main.txt")
    $evalOutputPath = Join-Path 'logs' ("l15_weight_4point_" + $preset + "_eval.txt")

    conda run -n mot python main.py *> $mainOutputPath
    conda run -n mot python evaluate_mota_idswitch.py *> $evalOutputPath

    Copy-Item results\virconv_OCM\car_summary.txt (Join-Path 'logs' ("car_summary_" + $preset + ".txt")) -Force
    Copy-Item results\virconv_OCM\car_detailed.csv (Join-Path 'logs' ("car_detailed_" + $preset + ".csv")) -Force
}

Remove-Item Env:L15_WEIGHT_PRESET -ErrorAction SilentlyContinue
