$ErrorActionPreference = "Stop"

$RepoRoot = "E:\mot"
$PythonExe = "E:\anaconda\envs\mot\python.exe"
$MainScript = Join-Path $RepoRoot "main.py"
$EvalScript = Join-Path $RepoRoot "evaluate_mota_idswitch.py"
$SummarySrc = Join-Path $RepoRoot "results\virconv_OCM\car_summary.txt"
$LogDir = Join-Path $RepoRoot "logs\boundary_hparam"
$SummaryCsv = Join-Path $LogDir "boundary_hparam_summary.csv"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

function Get-MetricMap {
    param(
        [string]$SummaryPath
    )
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

function Invoke-TrackedEval {
    param(
        [string]$Name,
        [hashtable]$EnvMap,
        [string]$GroupTag,
        [string]$ParamName,
        [string]$ParamValue
    )

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ("Running {0}" -f $Name) -ForegroundColor Green

    foreach ($key in $EnvMap.Keys) {
        $value = [string]$EnvMap[$key]
        Set-Item -Path ("Env:{0}" -f $key) -Value $value
        Write-Host ("  {0}={1}" -f $key, $value) -ForegroundColor Yellow
    }

    & $PythonExe $MainScript | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Tracking failed for $Name"
    }

    $resultDataDir = Join-Path $RepoRoot "results\virconv_OCM\data"
    if (Test-Path $resultDataDir) {
        Get-ChildItem -Path $resultDataDir -Filter *.txt | ForEach-Object {
            $cleanLines = Get-Content $_.FullName | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
            Set-Content -LiteralPath $_.FullName -Value $cleanLines
        }
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
        group = $GroupTag
        parameter = $ParamName
        value = $ParamValue
        experiment = $Name
        HOTA = $metricMap["HOTA"]
        IDSW = $metricMap["IDSW"]
        AssA = $metricMap["AssA"]
        IDF1 = $metricMap["IDF1"]
        Frag = $metricMap["Frag"]
        MOTA = $metricMap["MOTA"]
        summary_file = $dst
    }
}

$results = @()

# ============================================================
# Boundary-oriented hyperparameter analysis
# Goal:
#   not to find the single best point, but to identify
#   stable operating regions and degradation boundaries.
# ============================================================

# Experiment B-boundary:
# w_max in the motion budget allocation
#   w_ij^mot = w_min + (w_max - w_min) * tau_ij
# L4 takeover is disabled to isolate recovery-branch behavior.
$wMaxValues = @("0.10", "0.15", "0.20", "0.30", "0.40", "0.50", "0.60")
foreach ($val in $wMaxValues) {
    $results += Invoke-TrackedEval -Name ("BBoundary_wMax_{0}" -f $val.Replace('.', '_')) `
        -GroupTag "B-boundary" `
        -ParamName "w_max" `
        -ParamValue $val `
        -EnvMap @{
            ENABLE_L12_GEOM = "1"
            ENABLE_MOTION_REL = "1"
            ENABLE_L4_TAKEOVER = "0"
            L25_VEL_SHARE_MAX = $val
            L25_UNCERTAINTY_NORM = "12.0"
            L12_APPEARANCE_WEIGHT = "0.10"
            L15_ROTATED_GEOM_WEIGHT = "0.20"
        }
}

# Experiment C-boundary:
# c_h in the hits evidence activation
#   phi^hit = sigma(k_h * (h - c_h))
# L4 takeover is enabled and other takeover settings are fixed.
$hitsCenterValues = @(1, 2, 3, 4, 6, 8)
foreach ($val in $hitsCenterValues) {
    $results += Invoke-TrackedEval -Name ("CBoundary_cH_{0}" -f $val) `
        -GroupTag "C-boundary" `
        -ParamName "c_h" `
        -ParamValue ([string]$val) `
        -EnvMap @{
            ENABLE_L12_GEOM = "1"
            ENABLE_MOTION_REL = "1"
            ENABLE_L4_TAKEOVER = "1"
            L4_HANDOVER_HITS_CENTER = [string]$val
            L4_HANDOVER_SCORE_THRESHOLD = "0.72"
            L25_VEL_SHARE_MAX = "0.30"
            L25_UNCERTAINTY_NORM = "12.0"
            L12_APPEARANCE_WEIGHT = "0.10"
            L15_ROTATED_GEOM_WEIGHT = "0.20"
        }
}

$results | Export-Csv -Path $SummaryCsv -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "Boundary hyperparameter experiments finished." -ForegroundColor Green
Write-Host ("Summary CSV: {0}" -f $SummaryCsv) -ForegroundColor Cyan
$results | Format-Table -AutoSize
