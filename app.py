import gradio as gr
import os
import shutil
from src.vector_store import vector_store
from src.qa_chain import qa_chain

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

def chat_response(message, history):
    if not message:
        return "", history
    
    # Call QA Chain
    answer, _ = qa_chain.answer_question(message)
    
    history.append((message, answer))
    return "", history

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
        
    with gr.Tab("智能问答"):
        gr.Markdown("### 领域问答")
        chatbot = gr.Chatbot(label="对话记录", height=500)
        msg = gr.Textbox(label="请输入问题", placeholder="输入关于水声工程的问题...")
        clear = gr.Button("清空对话")
        
        msg.submit(chat_response, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    # Launch on all interfaces so it's accessible, but user said local offline.
    # 127.0.0.1 is default.
    demo.launch(server_name="127.0.0.1", server_port=7860)
