@echo off
chcp 65001
echo =================================================
echo       强力下载脚本 (使用 huggingface-cli)
echo =================================================

set "TARGET_DIR=e:\rag_project\models\bge-reranker-base"
set "HF_ENDPOINT=https://hf-mirror.com"

echo 正在清理旧的失败文件...
if exist "%TARGET_DIR%" (
    rmdir /s /q "%TARGET_DIR%"
    echo 旧目录已删除。
)

echo.
echo 正在启动下载...
echo 注意：如果进度条不动，请耐心等待，不要关闭窗口。
echo.

REM 尝试通过 Python 模块运行下载（兼容性最好）
python -m huggingface_hub.cli download --resume-download BAAI/bge-reranker-base --local-dir "%TARGET_DIR%" --local-dir-use-symlinks False

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [错误] 下载似乎失败了。
    echo 请截图报错信息给助手。
) else (
    echo.
    echo [成功] 模型已下载完成！
    echo 请重启 app.py 生效。
)
pause
