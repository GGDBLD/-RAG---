import os
import shutil
import stat
import sys
import time

# 1. 强制设置镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def handle_remove_readonly(func, path, exc):
    """Windows下删除只读文件的回调函数"""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        print(f"  无法删除文件 {path}: {e}")

def main():
    print("="*50)
    print("   RAG 模型最终版下载工具")
    print("="*50)
    
    # 检查库是否安装
    try:
        from huggingface_hub import snapshot_download
        print("✅ huggingface_hub 库已加载")
    except ImportError:
        print("❌ 错误: 未安装 huggingface_hub")
        print("请运行: pip install huggingface_hub")
        return

    target_dir = r"e:\rag_project\models\bge-reranker-base"
    model_id = "BAAI/bge-reranker-base"

    # 2. 清理旧目录
    if os.path.exists(target_dir):
        print(f"正在清理旧目录: {target_dir}")
        print("如果这里卡住，请手动关闭所有打开该文件夹的窗口...")
        try:
            shutil.rmtree(target_dir, onerror=handle_remove_readonly)
            print("清理完成。")
        except Exception as e:
            print(f"⚠️ 清理部分失败: {e}")
            print("尝试直接覆盖下载...")

    # 3. 开始下载
    print(f"\n正在从 {os.environ['HF_ENDPOINT']} 下载 {model_id} ...")
    print("目标路径:", target_dir)
    print("请耐心等待，不要关闭窗口...")

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False, # Windows 推荐 False
            resume_download=True,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # 忽略非必要文件加速下载
        )
        print("\n" + "="*50)
        print("✅✅✅ 下载成功！SUCCESS！")
        print("="*50)
        print(f"模型已保存在: {target_dir}")
        print("您可以重启 app.py 了。")
        
    except Exception as e:
        print("\n" + "!"*50)
        print("❌ 下载失败")
        print(f"错误信息: {e}")
        print("!"*50)
        print("\n可能原因及建议：")
        print("1. 网络连接被阻断 -> 请检查是否有代理软件干扰")
        print("2. 镜像站暂时不可用 -> 稍后重试")
        print("3. 如果反复失败，请尝试浏览器手动下载：")
        print(f"   访问: https://hf-mirror.com/{model_id}/tree/main")
        print(f"   将文件放入: {target_dir}")

if __name__ == "__main__":
    main()
