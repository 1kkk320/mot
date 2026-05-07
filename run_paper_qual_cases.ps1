$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if ($env:CONDA_PREFIX -and (Test-Path (Join-Path $env:CONDA_PREFIX "python.exe"))) {
    $python = Join-Path $env:CONDA_PREFIX "python.exe"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $python = (Get-Command python).Source
} else {
    throw "No usable python interpreter found."
}

$tool = Join-Path $root "tools\export_paper_figures.py"
$outRoot = Join-Path $root "results\paper_figures_cases"

New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

Write-Host ""
Write-Host "================ Paper Qualitative Cases ================"
Write-Host "Output root: $outRoot"
Write-Host "Python: $python"
Write-Host "Conda env: $($env:CONDA_DEFAULT_ENV)"
Write-Host "========================================================"
Write-Host ""

function Run-Case {
    param(
        [string]$CaseName,
        [string]$Seq,
        [string]$Frames,
        [string]$Dataset = "train",
        [string]$OnlyIds = ""
    )

    $caseOut = Join-Path $outRoot $CaseName
    New-Item -ItemType Directory -Force -Path $caseOut | Out-Null

    $args = @(
        $tool,
        "--dataset", $Dataset,
        "--seq", $Seq,
        "--frames", $Frames,
        "--result-root", "results\virconv_OCM",
        "--compare-root", "results\kitti_baseline",
        "--output-dir", $caseOut,
        "--title", "Current",
        "--draw-2d",
        "--draw-3d"
    )

    if ($OnlyIds -ne "") {
        $args += @("--only-ids", $OnlyIds)
    }

    Write-Host "[Run] $CaseName | seq=$Seq | frames=$Frames"
    & $python $args
    if ($LASTEXITCODE -ne 0) {
        throw "Case failed: $CaseName (exit code $LASTEXITCODE)"
    }
    Write-Host "[Done] $CaseName"
    Write-Host ""
}

Write-Host "Case A: baseline failed, current stable"
Write-Host "Note: requires non-empty results\virconv_OCM\data\0011.txt"
Write-Host ""

Run-Case -CaseName "A_normal_failure_compare" -Seq "0011" -Frames "258-261"
Run-Case -CaseName "B_geom_success" -Seq "0001" -Frames "9-10"
Run-Case -CaseName "C_motionrel_success" -Seq "0001" -Frames "42"
Run-Case -CaseName "D_takeover_success" -Seq "0001" -Frames "211-224"

Write-Host "All requested qualitative cases finished."
