<#
Set up Spatially Aware Property Inspection on a Windows machine with an NVIDIA GPU, in one command.

    powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Bundle C:\path\to\spatial-inspection-models
    powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Bundle ... -Splats      # + photoreal splats

What it does (from the repository folder, safe to run again):
  1. .venv        Python 3.11: PyTorch 2.5.1 + CUDA 12.4, OpenCV 5, Open3D, transformers, the app (editable)
  2. .venv-splat  Python 3.10: PyTorch 2.4.1 + CUDA 12.4 and the prebuilt gsplat wheel (only with -Splats)
  3. models       from the bundle when given (OpenCV Zoo models, Depth Anything 3 BASE, Grounding DINO tiny,
                  sample videos, optional demo results); otherwise the OpenCV models are downloaded and the two
                  Hugging Face models download on first use
  4. a check of the GPU, OpenCV and PyTorch

Needs: uv (https://docs.astral.sh/uv/: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"), an NVIDIA driver
for CUDA 12.4 or newer, about 15 GB of disk. Without an NVIDIA GPU the OpenCV-only path still runs (CPU).
#>
param(
    [string]$Bundle = "",      # the extracted spatial-inspection-models folder (or its .zip)
    [string]$Results = "",     # optional: spatial-inspection-demo-results.zip (finished demo walkthroughs)
    [switch]$Splats            # also build the photoreal Gaussian-splat environment
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Run($exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "failed: $exe $($argv -join ' ')" }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is not installed. Install it with:  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  then open a new terminal.'
}

# ------------------------------------------------------------------ 1. main environment
Step "1/4  .venv (Python 3.11, PyTorch CUDA 12.4, OpenCV 5)"
if (-not (Test-Path .venv\Scripts\python.exe)) { Run uv @("venv", ".venv", "--python", "3.11") }
$Py = "$Repo\.venv\Scripts\python.exe"
Run uv @("pip", "install", "--python", $Py, "torch==2.5.1+cu124", "torchvision==0.20.1+cu124",
         "--index-url", "https://download.pytorch.org/whl/cu124")
Run uv @("pip", "install", "--python", $Py, "-r", "requirements.txt", "-r", "requirements-dev.txt")
Run uv @("pip", "install", "--python", $Py, "-e", ".")

# ------------------------------------------------------------------ 2. splat environment
if ($Splats) {
    Step "2/4  .venv-splat (Python 3.10, PyTorch 2.4.1, prebuilt gsplat)"
    if (-not (Test-Path .venv-splat\Scripts\python.exe)) { Run uv @("venv", ".venv-splat", "--python", "3.10") }
    $SPy = "$Repo\.venv-splat\Scripts\python.exe"
    Run uv @("pip", "install", "--python", $SPy, "torch==2.4.1+cu124", "--index-url", "https://download.pytorch.org/whl/cu124")
    Run uv @("pip", "install", "--python", $SPy, "-r", "requirements-splat.txt",
             "--extra-index-url", "https://docs.gsplat.studio/whl/pt24cu124", "--index-strategy", "unsafe-best-match")
} else { Step "2/4  .venv-splat skipped (add -Splats for the photoreal mode)" }

# ------------------------------------------------------------------ 3. models
Step "3/4  models"
if ($Bundle) {
    if ($Bundle -like "*.zip") {
        $dest = Join-Path (Split-Path -Parent $Bundle) "spatial-inspection-models"
        if (-not (Test-Path $dest)) { Expand-Archive -Path $Bundle -DestinationPath $dest }
        $Bundle = $dest
    }
    if (-not (Test-Path "$Bundle\models\opencv")) { throw "not a model bundle: $Bundle (no models\opencv inside)" }
    New-Item -ItemType Directory -Force models\opencv, data\samples | Out-Null
    Copy-Item "$Bundle\models\opencv\*" models\opencv -Force
    if (Test-Path "$Bundle\data\samples") { Copy-Item "$Bundle\data\samples\*" data\samples -Force }
    # Hugging Face models go to the default cache, where from_pretrained() looks for them
    $hub = if ($env:HF_HOME) { Join-Path $env:HF_HOME "hub" } else { Join-Path $env:USERPROFILE ".cache\huggingface\hub" }
    New-Item -ItemType Directory -Force $hub | Out-Null
    Copy-Item "$Bundle\hf\hub\*" $hub -Recurse -Force
    Write-Host "  OpenCV models -> models\opencv;  Depth Anything 3 + Grounding DINO -> $hub"
} else {
    Run $Py @("-m", "spatial_inspection.models", "download")
    Write-Host "  Depth Anything 3 BASE and Grounding DINO tiny download from Hugging Face on first use (about 1.2 GB)"
}
if ($Results) {
    New-Item -ItemType Directory -Force results | Out-Null
    Expand-Archive -Path $Results -DestinationPath results -Force
    Write-Host "  demo walkthroughs -> results\"
}

# ------------------------------------------------------------------ 4. check
Step "4/4  check"
Run $Py @("-c", "import cv2, torch, open3d, transformers; print('OpenCV', cv2.__version__, '| PyTorch', torch.__version__, '| CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')")
if ($Splats) { Run "$Repo\.venv-splat\Scripts\python.exe" @("-c", "import torch, gsplat; print('gsplat', gsplat.__version__, '| CUDA', torch.cuda.is_available())") }

Write-Host @"

Ready. Start the web app and open http://localhost:8000 :
    .venv\Scripts\python.exe -m uvicorn spatial_inspection.server:app --port 8000
Inspect a video from the command line (one heavy job at a time; 2 threads by default):
    .venv\Scripts\python.exe -m spatial_inspection.walkthrough --video data\samples\tom_home.mp4 --name tom_home
"@ -ForegroundColor Green
