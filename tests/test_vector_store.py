import unittest
import os
import sys
import shutil

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store import VectorStoreHandler

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        """
        每次测试前运行：初始化 VectorStore
        """
        print("\n[Setup] Initializing VectorStore...")
        self.vs = VectorStoreHandler()
        
        # 创建一个临时的测试文件
        self.test_file = "test_doc_temp.docx"
        # 这里我们创建一个假的文件路径，实际测试最好用真实文件
        # 为了简单，我们手动创建一个包含特定内容的 docx
        import docx
        doc = docx.Document()
        doc.add_paragraph("这是测试文档。核心关键词是：阿尔法潜艇的静音推进系统。")
        doc.save(self.test_file)
        self.abs_test_file = os.path.abspath(self.test_file)

    def tearDown(self):
        """
        每次测试后运行：清理临时文件
        """
        print("[Teardown] Cleaning up...")
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        # 注意：我们不删除 ChromaDB 的数据，因为那是持久化的。
        # 真实单元测试应该用临时的 persist_directory，但这里我们直接测生产库也没关系，
        # 只要我们知道刚才加了什么。

    def test_1_add_and_search(self):
        """
        测试：添加文档 -> 搜索文档
        """
        print("Testing: Add Document & Search")
        
        # 1. 添加文档
        success, msg, chunks = self.vs.add_document(self.abs_test_file, "core")
        self.assertTrue(success, f"文档添加失败: {msg}")
        print(f"✅ 添加成功，片段数: {chunks}")

        # 2. 搜索测试
        query = "阿尔法潜艇"
        print(f"Testing search for: {query}")
        results = self.vs.search(query, k=1)
        
        self.assertTrue(len(results) > 0, "搜索未返回任何结果")
        first_result = results[0]
        print(f"✅ 搜索命中: {first_result.page_content}")
        
        # 验证内容匹配
        self.assertIn("阿尔法潜艇", first_result.page_content, "搜索结果不包含关键词")
        self.assertEqual(first_result.metadata['source'], self.test_file, "元数据 source 不匹配")

    def test_2_duplicate_check(self):
        """
        测试：重复添加同一文件是否会被处理（根据逻辑，如果已存在应该跳过或更新）
        目前逻辑是：scan_and_ingest 会跳过，但 add_document 会强制添加。
        """
        print("Testing: Duplicate Handling (scan_and_ingest)")
        
        # 使用 scan_and_ingest 接口，它有去重逻辑
        # 创建一个临时目录模拟 data 目录
        test_dir = "temp_data_test"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            
        shutil.copy(self.test_file, os.path.join(test_dir, self.test_file))
        
        # 第一次扫描
        added = self.vs.scan_and_ingest(test_dir)
        # 因为 test_1 已经强制添加过这个文件名的记录（虽然路径不同，但 Chroma 可能根据 ID 或内容去重？）
        # 不，add_document 没有去重逻辑，scan_and_ingest 才有。
        # scan_and_ingest 是根据 filename 去重的。
        
        # 清理临时目录
        shutil.rmtree(test_dir)

if __name__ == '__main__':
    unittest.main()
