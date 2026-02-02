import os
import shutil

def clear_chroma_db(collection_name: str = "water_acoustic_kb"):
    """
    清空向量库（测试用）
    """
    from src.vector_store import client
    try:
        client.delete_collection(name=collection_name)
        print(f"向量库集合 {collection_name} 已删除")
    except:
        print("向量库集合不存在，无需删除")
    
    # 清空文件夹
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        os.makedirs("./chroma_db")
        print("ChromaDB文件夹已清空")

def list_all_docs(data_dir: str = "./data") -> list:
    """
    列出所有文档文件
    """
    doc_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.docx', '.pdf')):
                doc_files.append(os.path.join(root, file))
    return doc_files

if __name__ == "__main__":
    # 测试工具函数
    print("所有文档：", list_all_docs())
    # clear_chroma_db()  # 谨慎使用，会清空知识库