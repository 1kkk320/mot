$ErrorActionPreference = "Stop"

$RepoRoot = "E:\mot"
$PythonExe = "E:\anaconda\envs\mot\python.exe"
$MainScript = Join-Path $RepoRoot "main.py"
$EvalScript = Join-Path $RepoRoot "evaluate_mota_idswitch.py"
$SummarySrc = Join-Path $RepoRoot "results\virconv_OCM\car_summary.txt"
$LogDir = Join-Path $RepoRoot "logs\l12_triplet_geom_weight"
$SummaryCsv = Join-Path $LogDir "l12_triplet_geom_weight_summary.csv"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

function Get-MetricMap {
    param([string]$SummaryPath)
    $lines = Get-Content $SummaryPath | Select-Object -First 2
    if ($lines.Count -lt 2) {
        throw "Invalid summary file: $SummaryPath"
    }
    $headers = ($lines[0] -split '\s+') | Where-Object { $_ -ne "" }
    $values = ($lines[1] -split '\s+') | Where-Object { $_ -ne "" }
    if ($headers.Count -ne $values.Count) {
        throw "Header/value length mismatch in $SummaryPath"
    }
    $map = @{}
    for ($i = 0; $i -lt $headers.Count; $i++) {
        $map[$headers[$i]] = $values[$i]
    }
    return $map
}

function Invoke-GeomRun {
    param(
        [string]$Name,
        [double]$BevWeight,
        [double]$CenterWeight,
        [double]$SizeWeight
    )

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ("Running {0}" -f $Name) -ForegroundColor Green
    Write-Host ("  weights = ({0:F2}, {1:F2}, {2:F2})" -f $BevWeight, $CenterWeight, $SizeWeight) -ForegroundColor Yellow

    $env:ENABLE_L12_GEOM = "1"
    $env:ENABLE_MOTION_REL = "1"
    $env:ENABLE_L4_TAKEOVER = "1"
    $env:L12_ROT_GEOM_BEV_WEIGHT = [string]$BevWeight
    $env:L12_ROT_GEOM_CENTER_WEIGHT = [string]$CenterWeight
    $env:L12_ROT_GEOM_SIZE_WEIGHT = [string]$SizeWeight
    $env:L12_ROT_GEOM_CENTER_TAU = "1.0"
    $env:L12_ROT_GEOM_SIZE_TAU = "1.0"

    & $PythonExe $MainScript | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Tracking failed for $Name"
    }

    & $PythonExe $EvalScript | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed for $Name"
    }

    if (-not (Test-Path $SummarySrc)) {
        throw "Missing summary: $SummarySrc"
    }

    $dst = Join-Path $LogDir ("car_summary_{0}.txt" -f $Name)
    Copy-Item -LiteralPath $SummarySrc -Destination $dst -Force
    $metricMap = Get-MetricMap -SummaryPath $dst

    [PSCustomObject]@{
        experiment = $Name
        lambda_bev = ("{0:F2}" -f $BevWeight)
        lambda_center = ("{0:F2}" -f $CenterWeight)
        lambda_size = ("{0:F2}" -f $SizeWeight)
        HOTA = $metricMap["HOTA"]
        AssA = $metricMap["AssA"]
        IDF1 = $metricMap["IDF1"]
        IDSW = $metricMap["IDSW"]
        MOTA = $metricMap["MOTA"]
        summary_file = $dst
    }
}

$results = @()
$results += Invoke-GeomRun -Name "bev_020_center_040_size_040" -BevWeight 0.20 -CenterWeight 0.40 -SizeWeight 0.40
$results += Invoke-GeomRun -Name "bev_033_center_033_size_033" -BevWeight 0.333333 -CenterWeight 0.333333 -SizeWeight 0.333333
$results += Invoke-GeomRun -Name "bev_050_center_025_size_025" -BevWeight 0.50 -CenterWeight 0.25 -SizeWeight 0.25
$results += Invoke-GeomRun -Name "bev_060_center_020_size_020" -BevWeight 0.60 -CenterWeight 0.20 -SizeWeight 0.20

$results | Export-Csv -Path $SummaryCsv -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "Triplet geometry weight analysis finished." -ForegroundColor Green
Write-Host ("Summary CSV: {0}" -f $SummaryCsv) -ForegroundColor Cyan
$results | Format-Table -AutoSize
