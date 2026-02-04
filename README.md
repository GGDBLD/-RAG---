# 水声工程 RAG 知识库系统

这是一个基于检索增强生成 (RAG) 的本地离线知识库系统，专为水声工程领域设计。系统支持多格式文档上传、解析、向量化存储，并结合本地大模型进行精准问答。

## 核心功能

*   **文档处理**: 支持 Word (.docx), 文本 PDF, 扫描版 PDF (自动 OCR)。
*   **向量检索**: 使用 ChromaDB 和 BAAI/bge-small-zh-v1.5 模型。
*   **智能问答**: 使用 Ollama 运行 Qwen-1.8B-Chat 模型，基于上下文回答并标注来源。
*   **交互界面**: Gradio Web 界面，支持文档上传和问答。

## 环境配置

### 1. Python 环境
建议使用 Anaconda 创建独立的虚拟环境 (Python 3.10):

```bash
conda create -n rag_env python=3.10
conda activate rag_env
```

安装项目依赖:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

*注意: PaddleOCR 依赖 `paddlepaddle`。如果您的机器只有 CPU，`pip install paddlepaddle` 即可。如果有 GPU 且想利用（虽然本项目默认配置为 CPU），请参考 PaddlePaddle 官网安装对应 CUDA 版本的包。*

### 2. Ollama 配置 (大模型)
确保本机已安装 [Ollama](https://ollama.com/) 并配置了环境变量 (通常安装程序会自动配置)。

启动 Ollama 服务并拉取 Qwen-1.8B-Chat 模型:

```bash
# 拉取模型 (需联网一次)
ollama pull qwen:1.8b-chat

# 启动服务 (默认运行在 localhost:11434)
ollama serve
```

### 3. 运行系统

在项目根目录下运行:

```bash
python app.py
```

启动后，浏览器会自动打开 `http://127.0.0.1:7860`。

## 目录结构

*   `app.py`: 程序入口，Gradio 前端。
*   `src/`: 核心源码
    *   `document_processing.py`: 文档解析 (Docx, PDF, OCR)。
    *   `vector_store.py`: 向量库管理 (ChromaDB)。
    *   `qa_chain.py`: 问答逻辑 (LangChain + Ollama)。
    *   `utils.py`: 通用工具。
*   `chroma_db/`: 向量库持久化目录 (自动生成)。

## 注意事项

*   **扫描版 PDF**: 系统会自动检测 PDF 是否为扫描版 (文本长度 < 10)，并调用 PaddleOCR 进行识别。首次运行 OCR 可能需要下载模型文件。
*   **离线运行**: 首次运行需要联网下载 Embedding 模型 (BGE) 和 OCR 模型。之后可完全离线运行。
