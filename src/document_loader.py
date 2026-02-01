import os
from pathlib import Path
from typing import List
import logging

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WaterAcousticsDocumentLoader:
    """水声领域多格式文档加载器"""
    
    def __init__(self, data_dir: str = "../data"):
        """
        初始化文档加载器
        
        参数:
            data_dir: 存放水声领域文档的目录路径
        """
        self.data_dir = Path(data_dir)
        self.supported_extensions = {'.pdf', '.docx', '.txt', '.md'}
        
    def load_documents(self) -> List[Document]:
        """加载data目录下的所有支持格式文档"""
        if not self.data_dir.exists():
            logger.error(f"数据目录不存在: {self.data_dir}")
            return []
            
        all_documents = []
        
        # 遍历数据目录
        for file_path in self.data_dir.rglob("*"):
            if file_path.suffix.lower() in self.supported_extensions:
                try:
                    docs = self._load_single_file(file_path)
                    all_documents.extend(docs)
                    logger.info(f"成功加载: {file_path.name} ({len(docs)}页/段)")
                except Exception as e:
                    logger.error(f"加载失败 {file_path.name}: {e}")
        
        logger.info(f"总计加载 {len(all_documents)} 个文档片段")
        return all_documents
    
    def _load_single_file(self, file_path: Path) -> List[Document]:
        """根据文件类型选择合适的加载器"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif suffix == '.docx':
            loader = Docx2txtLoader(str(file_path))
        elif suffix in ['.txt', '.md']:
            loader = TextLoader(str(file_path), encoding='utf-8')
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
            
        return loader.load()

# ===== 测试代码 =====
if __name__ == "__main__":
    print("=== 水声领域文档加载器测试 ===")
    print("当前工作目录:", os.getcwd())
    
    # 1. 创建测试数据目录（如果不存在）
    test_data_dir = Path("../data/test_samples")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 创建测试文件（实际项目中用你的真实水声文档替换）
    test_file = test_data_dir / "water_acoustics_intro.txt"
    test_content = """水声学(Hydroacoustics)是研究水下声波的产生、传播、接收和处理的学科。
主要应用领域包括：
1. 声纳技术(Sonar) - 用于水下目标探测、导航和通信
2. 海洋地球物理勘探
3. 水下通信系统
4. 海洋环境监测

关键技术参数：
- 频率范围：通常10Hz ~ 1MHz
- 声速：海水中约1500m/s
- 衰减系数：随频率增加而增大"""
    
    test_file.write_text(test_content, encoding='utf-8')
    print(f"创建测试文件: {test_file}")
    
    # 3. 测试加载器
    loader = WaterAcousticsDocumentLoader(data_dir="../data/test_samples")
    documents = loader.load_documents()
    
    if documents:
        print(f"\n✅ 成功加载 {len(documents)} 个文档片段")
        print("第一个片段预览:")
        print("-" * 50)
        print(documents[0].page_content[:200] + "...")
        print("-" * 50)
    else:
        print("\n❌ 文档加载失败，请检查数据目录")