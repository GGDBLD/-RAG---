import gradio as gr
import os
from src.vector_store import add_doc_to_vector_db
from src.qa_chain import get_answer

# 自定义CSS，优化界面
CSS = """
.gradio-container {max-width: 1200px !important;}
#chatbot {height: 500px !important;}
#upload_btn {margin-top: 10px !important;}
"""

def upload_doc(file, doc_type):
    """
    上传文档并入库
    """
    if not file:
        return "请选择要上传的文件！"
    
    try:
        # 入库
        chunk_count = add_doc_to_vector_db(file.name, doc_type)
        return f"文档上传成功！\n文件名：{os.path.basename(file.name)}\n新增片段数：{chunk_count}"
    except Exception as e:
        return f"文档上传失败：{str(e)}"

def chat(message, history):
    """
    问答交互逻辑
    """
    history = history or []
    if not message.strip():
        history.append((message, "请输入有效的问题！"))
        return history, history
    
    # 获取回答
    answer = get_answer(message)
    history.append((message, answer))
    return history, history

# 构建Gradio界面
with gr.Blocks(css=CSS, title="水声工程知识库") as demo:
    gr.Markdown("# 水声工程RAG知识库")
    gr.Markdown("### 步骤1：上传文档构建知识库 | 步骤2：输入问题获取回答")
    
    with gr.Tab("文档上传"):
        file_upload = gr.File(label="选择文档（支持.docx/.pdf，扫描版自动OCR）", file_types=[".docx", ".pdf"])
        doc_type = gr.Radio(["core", "supplement"], label="文档类型", value="core", 
                           info="core：核心教材 | supplement：补充文档")
        upload_btn = gr.Button("上传并入库", id="upload_btn")
        upload_output = gr.Textbox(label="上传结果", lines=3)
    
    with gr.Tab("智能问答"):
        chatbot = gr.Chatbot(id="chatbot", label="问答记录")
        msg = gr.Textbox(label="输入你的问题（如：海水声速与哪些因素有关？）")
        clear = gr.Button("清空对话")
        
        # 绑定事件
        msg.submit(chat, [msg, chatbot], [chatbot, chatbot])
        clear.click(lambda: (None, None), None, [chatbot, chatbot])
    
    # 绑定上传事件
    upload_btn.click(upload_doc, [file_upload, doc_type], upload_output)

# 初始化向量库（确保文件夹存在）
if not os.path.exists("./chroma_db"):
    os.makedirs("./chroma_db")

# 启动应用
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 本地运行，不生成公网链接
        inbrowser=True  # 自动打开浏览器
    )