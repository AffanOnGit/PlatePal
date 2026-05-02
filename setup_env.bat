@echo off
REM ═══════════════════════════════════════════════════════════════
REM  PlatePal — One-Shot Environment Setup (Windows + CUDA)
REM ═══════════════════════════════════════════════════════════════
REM
REM  This script:
REM    1. Creates a Python virtual environment
REM    2. Installs PyTorch with CUDA support
REM    3. Installs all project dependencies
REM    4. Installs OpenAI CLIP from GitHub
REM    5. Creates required data directories
REM    6. Verifies GPU detection
REM
REM  Usage:  setup_env.bat
REM ═══════════════════════════════════════════════════════════════

echo.
echo  ╔═══════════════════════════════════════╗
echo  ║   PlatePal Environment Setup          ║
echo  ╚═══════════════════════════════════════╝
echo.

REM ── Step 1: Create virtual environment ──────────────────────
if not exist "venv" (
    echo [1/6] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/6] Virtual environment already exists, skipping.
)

REM ── Step 2: Activate venv ───────────────────────────────────
echo [2/6] Activating virtual environment...
call venv\Scripts\activate.bat

REM ── Step 3: Install PyTorch with CUDA (Blackwell sm_120) ────
echo [3/6] Installing PyTorch with CUDA 12.8 support (RTX 50-series)...
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

REM ── Step 4: Install project dependencies ────────────────────
echo [4/6] Installing project dependencies...
pip install -r requirements.txt

REM ── Step 5: Install OpenAI CLIP ─────────────────────────────
echo [5/6] Installing OpenAI CLIP from GitHub...
pip install git+https://github.com/openai/CLIP.git

REM ── Step 6: Create data directories ─────────────────────────
echo [6/6] Creating data directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "checkpoints\text_model" mkdir checkpoints\text_model
if not exist "checkpoints\dcgan\samples" mkdir checkpoints\dcgan\samples
if not exist "evaluation" mkdir evaluation
if not exist "scripts" mkdir scripts

REM ── Verify GPU ──────────────────────────────────────────────
echo.
echo ════════════════════════════════════════
echo  GPU Verification
echo ════════════════════════════════════════
REM Allow 50-series to run despite the warning
set CUDA_MODULE_LOADING=LAZY
python -c "import torch; print(f'  CUDA Available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0)}'); print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB'); print(f'  CUDA Version: {torch.version.cuda}')"

if %ERRORLEVEL% neq 0 (
    echo.
    echo  ╔═══════════════════════════════════════╗
    echo  ║  ERROR: No CUDA GPU detected!         ║
    echo  ║  Training requires an NVIDIA GPU.     ║
    echo  ╚═══════════════════════════════════════╝
    pause
    exit /b 1
)

echo.
echo  ╔═══════════════════════════════════════╗
echo  ║  Setup Complete!                      ║
echo  ╚═══════════════════════════════════════╝
echo.
echo  Next steps:
echo    1. Download datasets into data\raw\
echo    2. Run: python scripts\preprocess.py
echo    3. Run: python train_text_model.py --corpus data\processed\recipes_train.txt --output checkpoints\text_model --fp16
echo    4. Run: python train_image_model.py --food101-h5 data\raw\food_c101_n10099_r64x64x3.h5 --cafd-dir data\raw\CAFD --output checkpoints\dcgan --fp16
echo.
pause

