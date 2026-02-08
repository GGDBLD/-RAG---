import sys
import os
import time

print("Step 1: Start", flush=True)

try:
    print("Step 2: Import Chroma", flush=True)
    from langchain_community.vectorstores import Chroma
    print("Step 2: OK", flush=True)
except Exception as e:
    print(f"Step 2 Error: {e}", flush=True)

try:
    print("Step 3: Import HuggingFaceEmbeddings", flush=True)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("Step 3: OK", flush=True)
except Exception as e:
    print(f"Step 3 Error: {e}", flush=True)

try:
    print("Step 4: Load Embedding Model (This is heavy)", flush=True)
    model_path = r"e:\rag_project\models\bge-small-zh-v1.5"
    if os.path.exists(model_path):
        print(f"Model path exists: {model_path}", flush=True)
        # 尝试加载模型
        # embedding = HuggingFaceEmbeddings(model_name=model_path)
        # print("Model Loaded OK", flush=True)
    else:
        print(f"Model path NOT FOUND: {model_path}", flush=True)
except Exception as e:
    print(f"Step 4 Error: {e}", flush=True)

print("Done.", flush=True)