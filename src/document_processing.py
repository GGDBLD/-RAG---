import os
import re
from docx import Document
from pypdf import PdfReader
from paddleocr import PaddleOCR
from typing import List, TextIO

# 初始化PaddleOCR（CPU版，禁用GPU）
try:
    # print("Creating PaddleOCR instance...", flush=True)
    # show_log参数在旧版本有效，新版本可能已移除或改名，这里移除以确保兼容性
    # use_angle_cls已废弃，建议使用use_angle_cls=True或ignore it? Warning said use `use_textline_orientation`.
    # Let's check the warning again: "The parameter `use_angle_cls` has been deprecated ... Please use `use_textline_orientation` instead."
    # But PaddleOCR signature might still accept use_angle_cls.
    # The error was "Unknown argument: show_log" and "Unknown argument: use_gpu".
    # Removing use_gpu as well.
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')
    # print("PaddleOCR initialized successfully.", flush=True)
except Exception as e:
    print(f"PaddleOCR initialization failed: {e}")
    ocr = None

def clean_text(text: str) -> str:
    """
    清洗文本：去除多余空格、换行、特殊字符
    """
    # 去除多余空格和换行
    text = re.sub(r'\s+', ' ', text)
    # 去除特殊字符（保留中文、英文、数字、标点）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''（）【】《》、·]', '', text)
    return text.strip()

def read_docx(file_path: str) -> str:
    """
    读取Word文档（.docx）
    """
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        return clean_text('\n'.join(full_text))
    except Exception as e:
        print(f"读取Word文档失败：{e}")
        return ""

def read_pdf_text(file_path: str) -> str:
    """
    读取PDF文本（可复制的文本PDF）
    """
    try:
        reader = PdfReader(file_path)
        full_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text.append(page_text)
        return clean_text('\n'.join(full_text))
    except Exception as e:
        print(f"读取PDF文本失败：{e}")
        return ""

def read_pdf_ocr(file_path: str) -> str:
    """
    OCR识别扫描版PDF（图片型PDF）
    """
    try:
        print(f"检测到扫描版PDF，启动OCR处理：{os.path.basename(file_path)}")
        # PaddleOCR直接处理PDF文件
        result = ocr.ocr(file_path, cls=True)
        full_text = []
        for page_result in result:
            if page_result:
                page_text = ' '.join([line[1][0] for line in page_result])
                full_text.append(page_text)
        return clean_text('\n'.join(full_text))
    except Exception as e:
        print(f"OCR识别PDF失败：{e}")
        return ""

def read_file(file_path: str) -> str:
    """
    统一读取文件入口：自动判断文件类型，扫描版PDF自动OCR
    """
    if not os.path.exists(file_path):
        print(f"文件不存在：{file_path}")
        return ""
    
    # 获取文件后缀
    ext = os.path.splitext(file_path)[-1].lower()
    
    # 读取不同类型文件
    if ext == '.docx':
        return read_docx(file_path)
    elif ext == '.pdf':
        # 先尝试普通读取，若内容为空则OCR
        text = read_pdf_text(file_path)
        if len(text) < 10:  # 内容过短，判定为扫描版
            text = read_pdf_ocr(file_path)
        return text
    else:
        print(f"不支持的文件类型：{ext}")
        return ""

def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    文本分块：将长文本切分为固定大小的片段，用于向量化
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # 重叠部分，避免语义割裂
        start += chunk_size - chunk_overlap
    
    return chunks

if __name__ == "__main__":
    # 测试函数
    # 使用绝对路径或相对于项目根目录的路径
    test_pdf = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "supplement", "复杂海洋环境舰船水下噪声特性研究.pdf")
    # test_pdf = "../data/supplement/测试扫描版.pdf"
    
    if os.path.exists(test_pdf):
        print(f"正在读取测试文件：{test_pdf}")
        test_text = read_file(test_pdf)
        print(f"读取到的文本长度：{len(test_text)}")
        print(f"文本片段数：{len(split_text(test_text))}")
        if len(test_text) > 0:
            print(f"前100字符预览：{test_text[:100]}")
    else:
        print(f"测试文件不存在：{test_pdf}")