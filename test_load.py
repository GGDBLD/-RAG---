import sys
import os

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(r"e:\rag_project"))

try:
    from src.qa_chain import qa_chain
    print("QA Chain loaded successfully")
except Exception as e:
    print(f"Error loading QA Chain: {e}")
