import sys
import os
import chromadb
from chromadb.config import Settings

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def direct_inspect():
    print("=== ChromaDB 直接检查 (无 Embedding) ===", flush=True)
    
    persist_directory = r"e:\rag_project\chroma_db"
    
    if not os.path.exists(persist_directory):
        print(f"❌ 目录不存在: {persist_directory}")
        return

    try:
        # 直接连接 ChromaDB 客户端
        client = chromadb.PersistentClient(path=persist_directory)
        
        # 列出所有集合
        collections = client.list_collections()
        print(f"📚 发现 {len(collections)} 个集合: {[c.name for c in collections]}")
        
        target_col = "water_acoustic_kb"
        if target_col not in [c.name for c in collections]:
            print(f"❌ 也就是没有找到名为 '{target_col}' 的知识库集合！")
            return
            
        collection = client.get_collection(target_col)
        count = collection.count()
        print(f"✅ 集合 '{target_col}' 包含 {count} 条数据片段 (Chunks)")
        
        if count > 0:
            print("\n--- 随机抽样 2 条数据 ---")
            # 这里的 get 不需要 embedding function
            data = collection.get(limit=2)
            
            for i, (doc_id, content, meta) in enumerate(zip(data['ids'], data['documents'], data['metadatas'])):
                print(f"\n[样本 {i+1}]")
                print(f"ID: {doc_id}")
                print(f"来源: {meta.get('source')} (Page {meta.get('page')})")
                print(f"内容: {content[:100].replace(chr(10), ' ')}...")
        else:
            print("\n⚠️ 警告: 集合是空的！")
            
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    direct_inspect()