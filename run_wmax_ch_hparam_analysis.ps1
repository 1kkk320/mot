$ErrorActionPreference = "Stop"

$RepoRoot = "E:\mot"
$PythonExe = "E:\anaconda\envs\mot\python.exe"
$MainScript = Join-Path $RepoRoot "main.py"
$EvalScript = Join-Path $RepoRoot "evaluate_mota_idswitch.py"
$SummarySrc = Join-Path $RepoRoot "results\virconv_OCM\car_summary.txt"
$LogDir = Join-Path $RepoRoot "logs\wmax_ch_hparam"
$SummaryCsv = Join-Path $LogDir "wmax_ch_hparam_summary.csv"

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
        [hashtable]$EnvMap
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
# Experiment B: motion budget upper-bound analysis
# Corresponds to w_max in:
#   w_ij^mot = w_min + (w_max - w_min) * tau_ij
# L4 takeover is disabled to isolate the L1.5/L2.5 recovery branch.
# ============================================================
$wMaxValues = @("0.15", "0.30", "0.50")
foreach ($val in $wMaxValues) {
    $results += Invoke-TrackedEval -Name ("B_wMax_{0}" -f $val.Replace('.', '_')) -EnvMap @{
        ENABLE_L12_GEOM = "1"
        ENABLE_MOTION_REL = "1"
        ENABLE_L4_TAKEOVER = "0"
        L25_VEL_SHARE_MAX = $val
        L25_UNCERTAINTY_NORM = "12.0"
        L12_APPEARANCE_WEIGHT = "0.10"
        L15_ROTATED_GEOM_WEIGHT = "0.20"
    }
}

# ============================================================
# Experiment C: hits-center analysis for progressive takeover
# Corresponds to c_h in:
#   phi^hit = sigma(k_h * (h - c_h))
# Recovery branch remains enabled and L4 takeover is isolated via c_h.
# ============================================================
$hitsCenterValues = @(2, 4, 6)
foreach ($val in $hitsCenterValues) {
    $results += Invoke-TrackedEval -Name ("C_cH_{0}" -f $val) -EnvMap @{
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
Write-Host "w_max + c_h hyperparameter experiments finished." -ForegroundColor Green
Write-Host ("Summary CSV: {0}" -f $SummaryCsv) -ForegroundColor Cyan
$results | Format-Table -AutoSize
