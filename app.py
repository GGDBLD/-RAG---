import gradio as gr
import os
import shutil
from src.vector_store import vector_store
from src.qa_chain import qa_chain
try:
    from gradio import ChatMessage
except ImportError:
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
    folder_path = "data"
    if not os.path.exists(folder_path):
        return f"文件夹 {folder_path} 不存在。"
    added_files = vector_store.scan_and_ingest(folder_path)
    if added_files:
        return f"同步成功！已自动添加 {len(added_files)} 个新文件：\n" + "\n".join(added_files)
    else:
        return "data 文件夹中没有发现新文件（所有文件均已入库）。"


def get_kb_overview():
    try:
        files = vector_store.get_indexed_files()
        num_files = len(files)
        data = vector_store.vectordb.get()
        num_chunks = 0
        if data and "ids" in data:
            num_chunks = len(data["ids"])
        lines = [
            f"已入库文档数：{num_files}",
            f"向量片段总数：{num_chunks}",
        ]
        if files:
            preview = ", ".join(sorted(files)[:5])
            if len(files) > 5:
                preview += " ..."
            lines.append(f"示例文档：{preview}")
        else:
            lines.append("当前尚未入库任何文档。")
        return "\n".join(lines)
    except Exception as e:
        return f"获取知识库概览失败: {e}"

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

custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue"
)

with gr.Blocks(
    title="水声工程 RAG 知识库系统",
    theme=custom_theme,
) as demo:
    gr.Markdown("# 水声工程领域离线知识库系统")
    
    with gr.Tab("文档管理与入库"):
        gr.Markdown("### 文档管理与入库")
        with gr.Row():
            with gr.Column(scale=3):
                file_input = gr.File(
                    label="上传文件 (.docx, .pdf, .txt)",
                    file_types=[".docx", ".pdf", ".txt"]
                )
                doc_type_input = gr.Radio(
                    choices=["core", "supplement"],
                    label="文档类型",
                    value="core"
                )
                upload_button = gr.Button("📥 上传并入库", variant="primary")
            with gr.Column(scale=2):
                upload_output = gr.Textbox(
                    label="入库结果反馈",
                    interactive=False,
                    lines=6
                )
        
        upload_button.click(
            upload_and_process,
            inputs=[file_input, doc_type_input],
            outputs=upload_output
        )

        gr.Markdown("**批量同步 data 文件夹中文档**")
        with gr.Row():
            sync_button = gr.Button("🔄 扫描 data 文件夹并同步")
            sync_output = gr.Textbox(label="同步结果", interactive=False, lines=6)
        
        sync_button.click(
            sync_data_folder_ui,
            inputs=[],
            outputs=sync_output
        )

        gr.Markdown("**知识库概览**")
        with gr.Row():
            kb_button = gr.Button("📊 刷新知识库概览")
            kb_overview = gr.Textbox(label="知识库概览", interactive=False, lines=6)
        kb_button.click(
            get_kb_overview,
            inputs=[],
            outputs=kb_overview
        )
        
    with gr.Tab("智能问答"):
        gr.Markdown("### 领域智能问答（输入问题后按回车或点击“发送问题”）")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="对话记录", height=500)
            with gr.Column(scale=2):
                gr.Markdown("在下方输入问题，按回车或点击按钮即可发送。")

        with gr.Row():
            msg = gr.Textbox(
                label="请输入问题",
                placeholder="例如：多途效应对被动声纳信号处理有什么影响？",
                lines=2
            )
            send = gr.Button("发送问题", variant="primary")
            clear = gr.Button("🧹 清空对话")

        gr.Markdown("#### 示例问题（点击可自动填入输入框）")
        with gr.Row():
            ex_q1 = "水声工程的主要研究方向有哪些？"
            ex_q2 = "多途效应会如何影响声纳探测性能？"
            ex_q3 = "舰船水下噪声的主要来源和特点是什么？"
            ex1 = gr.Button("示例 1：研究方向")
            ex2 = gr.Button("示例 2：多途效应")
            ex3 = gr.Button("示例 3：水下噪声")
        ex1.click(lambda q=ex_q1: q, inputs=None, outputs=msg)
        ex2.click(lambda q=ex_q2: q, inputs=None, outputs=msg)
        ex3.click(lambda q=ex_q3: q, inputs=None, outputs=msg)

        send.click(chat_response, [msg, chatbot], [msg, chatbot])
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
