@echo off
echo ==========================================
echo       Downloading Reranker Model
echo ==========================================

REM 1. Create directory if not exists
if not exist "e:\rag_project\models" (
    echo Creating models directory...
    mkdir "e:\rag_project\models"
)

cd /d "e:\rag_project\models"

REM 2. Install Git LFS
echo Installing Git LFS...
git lfs install

REM 3. Clone repository
if exist "bge-reranker-base" (
    echo Directory bge-reranker-base already exists.
    echo Please delete it if you want to re-download.
) else (
    echo Cloning from hf-mirror.com...
    git clone https://hf-mirror.com/BAAI/bge-reranker-base
)

echo.
echo ==========================================
echo Download process finished.
echo Please check for any error messages above.
echo ==========================================
pause
