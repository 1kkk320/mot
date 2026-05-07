$ErrorActionPreference = "Stop"

$RepoRoot = "E:\mot"
$PythonExe = "E:\anaconda\envs\mot\python.exe"
$MainScript = Join-Path $RepoRoot "main.py"
$EvalScript = Join-Path $RepoRoot "evaluate_mota_idswitch.py"
$SummarySrc = Join-Path $RepoRoot "results\virconv_OCM\car_summary.txt"
$LogDir = Join-Path $RepoRoot "logs\topjournal_hparam"
$SummaryCsv = Join-Path $LogDir "topjournal_hparam_summary.csv"

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
# Reliable anchor selection
# - B1 and D use the strongest no-regression mainline:
#   Geom + MotionRel, without L4 takeover.
# - C1 enables takeover because it specifically studies PIT.
# ============================================================

# B1: uncertainty normalization robustness
$b1Values = @(8, 12, 16)
foreach ($val in $b1Values) {
    $results += Invoke-TrackedEval -Name ("B1_kappaU_{0}" -f $val) -EnvMap @{
        ENABLE_L12_GEOM = "1"
        ENABLE_MOTION_REL = "1"
        ENABLE_L4_TAKEOVER = "0"
        L25_UNCERTAINTY_NORM = [string]$val
        L12_APPEARANCE_WEIGHT = "0.10"
        L15_ROTATED_GEOM_WEIGHT = "0.20"
        L25_VEL_SHARE_MAX = "0.30"
    }
}

# C1: takeover threshold sensitivity
$c1Values = @("0.65", "0.72", "0.80")
foreach ($val in $c1Values) {
    $results += Invoke-TrackedEval -Name ("C1_tauH_{0}" -f $val.Replace('.', '_')) -EnvMap @{
        ENABLE_L12_GEOM = "1"
        ENABLE_MOTION_REL = "1"
        ENABLE_L4_TAKEOVER = "1"
        L4_HANDOVER_SCORE_THRESHOLD = $val
        L25_UNCERTAINTY_NORM = "12.0"
        L12_APPEARANCE_WEIGHT = "0.10"
        L15_ROTATED_GEOM_WEIGHT = "0.20"
        L25_VEL_SHARE_MAX = "0.30"
    }
}

# D: Geom x MotionRel interaction
# Geom strength is approximated by how much the main association trusts geometry
# over appearance, together with the L1.5 rotated-geometry support.
$geomPresets = @(
    @{ tag = "weak";   app = "0.14"; l15geom = "0.10" },
    @{ tag = "medium"; app = "0.10"; l15geom = "0.20" },
    @{ tag = "strong"; app = "0.06"; l15geom = "0.30" }
)
$motionMaxValues = @("0.20", "0.30", "0.40")

foreach ($geom in $geomPresets) {
    foreach ($velMax in $motionMaxValues) {
        $results += Invoke-TrackedEval -Name ("D_{0}_velMax_{1}" -f $geom.tag, $velMax.Replace('.', '_')) -EnvMap @{
            ENABLE_L12_GEOM = "1"
            ENABLE_MOTION_REL = "1"
            ENABLE_L4_TAKEOVER = "0"
            L12_APPEARANCE_WEIGHT = $geom.app
            L15_ROTATED_GEOM_WEIGHT = $geom.l15geom
            L25_VEL_SHARE_MAX = $velMax
            L25_UNCERTAINTY_NORM = "12.0"
        }
    }
}

$results | Export-Csv -Path $SummaryCsv -NoTypeInformation -Encoding UTF8

Write-Host ""
Write-Host "All top-journal hyperparameter experiments finished." -ForegroundColor Green
Write-Host ("Summary CSV: {0}" -f $SummaryCsv) -ForegroundColor Cyan
$results | Format-Table -AutoSize
