$ErrorActionPreference = "Stop"

$RepoRoot = "E:\mot"
$PythonExe = "E:\anaconda\envs\mot\python.exe"
$SummarySrc = Join-Path $RepoRoot "results\virconv_OCM\car_summary.txt"
$LogDir = Join-Path $RepoRoot "logs"
$EvalScript = Join-Path $RepoRoot "evaluate_mota_idswitch.py"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$experiments = @(
    @{
        Name = "ablation_semantic_v3_geom"
        Env = @{
            ENABLE_L12_GEOM = "1"
            ENABLE_MOTION_REL = "0"
            ENABLE_L4_TAKEOVER = "0"
        }
    },
    @{
        Name = "ablation_semantic_v3_motionrel"
        Env = @{
            ENABLE_L12_GEOM = "0"
            ENABLE_MOTION_REL = "1"
            ENABLE_L4_TAKEOVER = "0"
        }
    },
    @{
        Name = "ablation_semantic_v3_l4_takeover"
        Env = @{
            ENABLE_L12_GEOM = "0"
            ENABLE_MOTION_REL = "0"
            ENABLE_L4_TAKEOVER = "1"
        }
    },
    @{
        Name = "ablation_semantic_v3_geom_motionrel"
        Env = @{
            ENABLE_L12_GEOM = "1"
            ENABLE_MOTION_REL = "1"
            ENABLE_L4_TAKEOVER = "0"
        }
    },
    @{
        Name = "ablation_semantic_v3_geom_takeover"
        Env = @{
            ENABLE_L12_GEOM = "1"
            ENABLE_MOTION_REL = "0"
            ENABLE_L4_TAKEOVER = "1"
        }
    },
    @{
        Name = "ablation_semantic_v3_motionrel_takeover"
        Env = @{
            ENABLE_L12_GEOM = "0"
            ENABLE_MOTION_REL = "1"
            ENABLE_L4_TAKEOVER = "1"
        }
    }
)

foreach ($exp in $experiments) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ("Running {0}" -f $exp.Name) -ForegroundColor Green
    Write-Host ("ENABLE_L12_GEOM={0} ENABLE_MOTION_REL={1} ENABLE_L4_TAKEOVER={2}" -f `
        $exp.Env.ENABLE_L12_GEOM, $exp.Env.ENABLE_MOTION_REL, $exp.Env.ENABLE_L4_TAKEOVER) -ForegroundColor Yellow

    $env:ENABLE_L12_GEOM = $exp.Env.ENABLE_L12_GEOM
    $env:ENABLE_MOTION_REL = $exp.Env.ENABLE_MOTION_REL
    $env:ENABLE_L4_TAKEOVER = $exp.Env.ENABLE_L4_TAKEOVER

    & $PythonExe "$RepoRoot\main.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed for $($exp.Name) with exit code $LASTEXITCODE"
    }

    & $PythonExe $EvalScript
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed for $($exp.Name) with exit code $LASTEXITCODE"
    }

    if (-not (Test-Path $SummarySrc)) {
        throw "Summary file not found: $SummarySrc"
    }

    $dst = Join-Path $LogDir ("car_summary_{0}.txt" -f $exp.Name)
    Copy-Item -LiteralPath $SummarySrc -Destination $dst -Force

    $lines = Get-Content $dst | Select-Object -First 2
    Write-Host ("Saved summary to {0}" -f $dst) -ForegroundColor Cyan
    $lines | ForEach-Object { Write-Host $_ }
}

Write-Host ""
Write-Host "All 6 paper-semantic ablations finished." -ForegroundColor Green
