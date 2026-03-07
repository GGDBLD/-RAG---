import os
import sys
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 数据库路径
DB_DIR = r"e:\rag_project\chroma_db"
EMBEDDING_MODEL_PATH = r"e:\rag_project\models\bge-large-zh-v1.5"

def inspect_chroma_content(file_name_keyword="计算海洋声学"):
    print("-" * 50)
    print(f"🔍 Inspecting ChromaDB for documents containing: '{file_name_keyword}'")
    print("-" * 50)

    # 1. 初始化 Embedding 模型
    if not os.path.exists(EMBEDDING_MODEL_PATH):
        print(f"❌ Embedding model not found at {EMBEDDING_MODEL_PATH}")
        return

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}")
        return

    # 2. 连接 ChromaDB
    try:
        vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        # 获取所有数据（Chroma API 可能变动，这里使用 get() 方法）
        collection_data = vectordb.get()
        
        ids = collection_data['ids']
        metadatas = collection_data['metadatas']
        documents = collection_data['documents']

        print(f"✅ Successfully connected to DB. Total chunks: {len(ids)}")
    except Exception as e:
        print(f"❌ Failed to connect to ChromaDB: {e}")
        return

    # 3. 过滤并展示特定文件的内容
    found_count = 0
    print("\n--- Previewing Content ---")
    
    for i, meta in enumerate(metadatas):
        source = meta.get('source', '')
        # 检查文件名是否包含关键词
        if file_name_keyword in source:
            found_count += 1
            print(f"\n📄 Chunk {found_count} from: {source} (Page {meta.get('page', '?')})")
            print("-" * 30)
            # 打印前 200 个字符预览
            content_preview = documents[i][:200].replace('\n', ' ')
            print(f"{content_preview}...")
            print("-" * 30)
            
            # 只展示前 5 个片段，避免刷屏
            if found_count >= 5:
                print("\n... (Stopped previewing after 5 chunks) ...")
                break
    
    if found_count == 0:
        print(f"\n❌ No documents found matching keyword '{file_name_keyword}'.")
        print("Possible reasons:")
        print("1. OCR is still running and hasn't finished ingesting.")
        print("2. The file name in DB metadata is different.")
        print("3. OCR failed completely and produced no text.")
    else:
        print(f"\n✅ Found total {found_count} chunks related to '{file_name_keyword}' (showing first 5).")

if __name__ == "__main__":
    # 可以通过命令行参数传入关键词
    keyword = sys.argv[1] if len(sys.argv) > 1 else "计算海洋声学"
    inspect_chroma_content(keyword)
