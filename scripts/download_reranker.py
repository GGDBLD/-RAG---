import os
from huggingface_hub import snapshot_download

# 设置镜像源，加速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 模型名称
model_name = "BAAI/bge-reranker-base"

# 本地保存路径
local_dir = r"e:\rag_project\models\bge-reranker-base"

print(f"开始下载 {model_name} 到 {local_dir} ...")
print("请耐心等待，取决于您的网速...")

try:
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # Windows下不使用软链接，直接下载文件
        resume_download=True
    )
    print("\n下载成功！")
    print(f"模型已保存在: {local_dir}")
    print("现在请重启您的 RAG 应用程序 (app.py) 以启用重排序功能。")
    
except Exception as e:
    print(f"\n下载失败: {str(e)}")
    print("如果自动下载失败，您可以尝试使用 Git 命令手动下载：")
    print(f"git clone https://hf-mirror.com/{model_name} {local_dir}")
