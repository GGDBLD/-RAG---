import gradio as gr
import os
import shutil
from src.vector_store import vector_store
from src.qa_chain import qa_chain
try:
    from gradio import ChatMessage
except ImportError:
    # Fallback for older Gradio versions
    class ChatMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content

def upload_and_process(file_obj, doc_type):
    if not file_obj:
        return "请选择文件。"
    
    # Save temp file to work with absolute path or keep it as is
    # Gradio passes a temp file path usually
    file_path = file_obj.name
    
    # Ensure we can read it. 
    # If the user wants to keep files in a specific directory, we might move them.
    # For now, just process the temp file.
    
    success, msg, num_chunks = vector_store.add_document(file_path, doc_type)
    
    if success:
        return f"成功！文件名: {os.path.basename(file_path)}\n类型: {doc_type}\n新增片段数: {num_chunks}"
    else:
        return f"失败: {msg}"

def sync_data_folder_ui():
    """UI wrapper for data folder sync"""
    folder_path = "data"
    if not os.path.exists(folder_path):
        return f"文件夹 {folder_path} 不存在。"
    
    added_files = vector_store.scan_and_ingest(folder_path)
    if added_files:
        return f"同步成功！已自动添加 {len(added_files)} 个新文件：\n" + "\n".join(added_files)
    else:
        return "data 文件夹中没有发现新文件（所有文件均已入库）。"

def chat_response(message, history):
    if not message:
        return "", history
    
    # Init history if needed
    if history is None:
        history = []

    # Call QA Chain
    answer, _ = qa_chain.answer_question(message, history)
    
    # Use ChatMessage objects to satisfy strict Gradio format requirements
    new_history = []
    
    # Convert existing history
    for item in history:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            # Convert old tuple format
            new_history.append(ChatMessage(role="user", content=str(item[0])))
            new_history.append(ChatMessage(role="assistant", content=str(item[1])))
        elif isinstance(item, dict):
            # Convert dict format
            new_history.append(ChatMessage(role=item.get("role"), content=item.get("content")))
        elif hasattr(item, 'role') and hasattr(item, 'content'):
            # Already ChatMessage
            new_history.append(item)
    
    # Append new interaction
    new_history.append(ChatMessage(role="user", content=message))
    new_history.append(ChatMessage(role="assistant", content=answer))
    
    return "", new_history

with gr.Blocks(title="水声工程 RAG 知识库系统") as demo:
    gr.Markdown("# 水声工程领域离线知识库系统")
    
    with gr.Tab("文档上传"):
        gr.Markdown("### 上传文档到知识库")
        with gr.Row():
            file_input = gr.File(
                label="上传文件 (.docx, .pdf)",
                file_types=[".docx", ".pdf"]
            )
            doc_type_input = gr.Radio(
                choices=["core", "supplement"],
                label="文档类型",
                value="core"
            )
        upload_button = gr.Button("上传并入库")
        upload_output = gr.Textbox(label="上传结果", interactive=False)
        
        upload_button.click(
            upload_and_process,
            inputs=[file_input, doc_type_input],
            outputs=upload_output
        )

        gr.Markdown("---")
        gr.Markdown("### 自动同步本地文件夹")
        gr.Markdown("将文件放入项目根目录下的 `data` 文件夹，点击下方按钮即可批量入库。")
        sync_button = gr.Button("🔄 扫描 data 文件夹并同步")
        sync_output = gr.Textbox(label="同步结果", interactive=False)
        
        sync_button.click(
            sync_data_folder_ui,
            inputs=[],
            outputs=sync_output
        )
        
    with gr.Tab("智能问答"):
        gr.Markdown("### 领域问答")
        # Gradio 3.x compatibility: Do not use 'type' argument. Defaults to tuples.
        # EXPLICITLY set type='messages' to match the error requirement?
        # No, user said type='messages' caused TypeError.
        # So we leave type unspecified, BUT we provide 'messages' format data because the component demands it at runtime.
        chatbot = gr.Chatbot(label="对话记录", height=500)

        msg = gr.Textbox(label="请输入问题", placeholder="输入关于水声工程的问题...")
        clear = gr.Button("清空对话")
        
        msg.submit(chat_response, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    # Auto-sync data folder on startup
    print("Startup: Scanning 'data' folder for new documents...")
    added = vector_store.scan_and_ingest("data")
    if added:
        print(f"Startup: Auto-ingested {len(added)} files from data folder.")
    else:
        print("Startup: No new files found in data folder.")

    # Launch on all interfaces so it's accessible, but user said local offline.
    # 127.0.0.1 is default.
    demo.launch(server_name="127.0.0.1", server_port=7860)
