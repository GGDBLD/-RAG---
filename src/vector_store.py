import os
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from src.document_processing import read_file, split_text

# 初始化ChromaDB客户端
CHROMA_DB_PATH = "./chroma_db"
client = Client(Settings(persist_directory=CHROMA_DB_PATH, anonymized_telemetry=False))

# 初始化嵌入模型（BGE中文模型）
embedding_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

# 创建/获取知识库集合
def get_or_create_collection(collection_name: str = "water_acoustic_kb"):
    """
    获取或创建ChromaDB集合
    """
    try:
        collection = client.get_collection(name=collection_name)
    except:
        collection = client.create_collection(name=collection_name)
    return collection

def text_to_embeddings(texts: List[str]) -> List[List[float]]:
    """
    将文本列表转换为向量列表
    """
    return embedding_model.encode(texts, convert_to_numpy=False).tolist()

def add_doc_to_vector_db(file_path: str, doc_type: str = "supplement"):
    """
    将文档读取、分块、向量化后存入向量库
    """
    # 1. 读取文件
    raw_text = read_file(file_path)
    if not raw_text:
        print(f"文件无有效内容，跳过：{file_path}")
        return
    
    # 2. 文本分块
    text_chunks = split_text(raw_text)
    if not text_chunks:
        print(f"文本分块失败，跳过：{file_path}")
        return
    
    # 3. 生成向量
    embeddings = text_to_embeddings(text_chunks)
    
    # 4. 准备元数据和ID
    file_name = os.path.basename(file_path)
    ids = [f"{file_name}_{i}" for i in range(len(text_chunks))]
    metadatas = [{"source": file_name, "type": doc_type} for _ in text_chunks]
    
    # 5. 存入向量库
    collection = get_or_create_collection()
    collection.add(
        documents=text_chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    
    print(f"文档入库成功：{file_name} → 新增 {len(text_chunks)} 个片段")
    return len(text_chunks)

def search_similar_text(query: str, top_k: int = 3) -> List[Dict]:
    """
    检索与查询相似的文本片段
    """
    # 生成查询向量
    query_embedding = embedding_model.encode([query])[0].tolist()
    
    # 检索相似片段
    collection = get_or_create_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # 格式化结果
    similar_docs = []
    for i in range(top_k):
        if i < len(results['documents'][0]):
            similar_docs.append({
                "text": results['documents'][0][i],
                "source": results['metadatas'][0][i]['source'],
                "type": results['metadatas'][0][i]['type']
            })
    
    return similar_docs

if __name__ == "__main__":
    # 测试向量库
    test_file = "../data/main/水声学基础.docx"
    add_doc_to_vector_db(test_file, "core")
    
    # 测试检索
    query = "海水声速与哪些因素有关？"
    results = search_similar_text(query)
    print(f"\n检索结果（{query}）：")
    for doc in results:
        print(f"来源：{doc['source']} → {doc['text'][:100]}...")