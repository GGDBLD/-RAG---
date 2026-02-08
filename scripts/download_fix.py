import os
import shutil
import subprocess
import sys

def main():
    # 1. 设置目标目录
    target_dir = r"e:\rag_project\models\bge-reranker-base"
    
    print("="*50)
    print("   RAG 模型修复下载工具")
    print("="*50)

    # 2. 清理旧文件 (防止之前的失败残留导致错误)
    if os.path.exists(target_dir):
        print(f"正在清理旧目录: {target_dir} ...")
        try:
            shutil.rmtree(target_dir)
            print("清理完成。")
        except Exception as e:
            print(f"清理失败 (可能是文件被占用): {e}")
            print("尝试继续下载...")

    # 3. 设置镜像环境变量
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"已设置镜像源: {os.environ['HF_ENDPOINT']}")

    # 4. 构造下载命令
    # 使用 python -m huggingface_hub.cli 避免路径问题
    cmd = [
        sys.executable, "-m", "huggingface_hub.cli", "download",
        "BAAI/bge-reranker-base",
        "--local-dir", target_dir,
        "--local-dir-use-symlinks", "False",
        "--resume-download"
    ]

    print("\n开始下载...")
    print(f"执行命令: {' '.join(cmd)}")
    print("-" * 50)

    # 5. 执行命令
    try:
        # 使用 subprocess 调用，可以看到实时进度条
        ret = subprocess.call(cmd)
        
        print("-" * 50)
        if ret == 0:
            print("✅ 下载成功！")
            print(f"模型已保存在: {target_dir}")
            print("请重启 app.py 生效。")
        else:
            print("❌ 下载失败。")
            print("可能原因：网络波动或镜像站暂时不稳定。")
            print("建议：请稍后再次运行此脚本重试。")
            
    except Exception as e:
        print(f"执行出错: {e}")

if __name__ == "__main__":
    main()
