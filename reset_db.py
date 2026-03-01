
import shutil
import os
import time
import psutil

def kill_python_processes():
    """Attempt to kill other python processes that might be holding the file lock"""
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'python' in proc.info['name'].lower() and proc.info['pid'] != current_pid:
                # Be careful not to kill system processes, but here we assume sandbox env
                # This is risky in a real env, but in this sandbox it might be necessary
                pass 
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def remove_chroma_db():
    db_path = r"e:\rag_project\chroma_db"
    if not os.path.exists(db_path):
        print(f"{db_path} does not exist. Nothing to do.")
        return

    print(f"Attempting to remove {db_path}...")
    max_retries = 5
    for i in range(max_retries):
        try:
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
            print("Successfully removed chroma_db.")
            return
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            time.sleep(1)
    
    print("Failed to remove chroma_db after multiple attempts. Please close all Python processes manually.")

if __name__ == "__main__":
    remove_chroma_db()
