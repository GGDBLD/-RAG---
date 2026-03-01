import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path regardless of where this script is run from
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules to avoid loading heavy models or missing dependencies
sys.modules['langchain_community.llms'] = MagicMock()
sys.modules['langchain_core.prompts'] = MagicMock()
sys.modules['langchain_core.documents'] = MagicMock()
sys.modules['src.vector_store'] = MagicMock()
sys.modules['src.utils'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['langchain_huggingface'] = MagicMock()
sys.modules['langchain_community.embeddings'] = MagicMock()
sys.modules['langchain_community.vectorstores'] = MagicMock()

# Mocking Document class specifically since it's imported
class MockDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

sys.modules['langchain_core.documents'].Document = MockDocument

# Now import the class to test
# We need to suppress the singleton instantiation at the end of qa_chain.py
# or just let it happen with mocks.
from src.qa_chain import QAChainHandler

class TestQAChainStreaming(unittest.TestCase):
    def setUp(self):
        # Patch os.path.exists to avoid real file checks
        self.path_patcher = patch('os.path.exists')
        self.mock_exists = self.path_patcher.start()
        self.mock_exists.return_value = False # Pretend reranker doesn't exist to skip loading
        
        self.handler = QAChainHandler()
        # Re-mock LLM and components ensuring they are ready for test
        self.handler.llm = MagicMock()
        self.handler.prompt = MagicMock()
        self.handler.reranker = None
        
    def tearDown(self):
        self.path_patcher.stop()
        
    def test_streaming_generator(self):
        print("\nTesting Streaming Generator...")
        # Mock retrieval context
        mock_doc = MockDocument("Test content", {"source": "test.pdf", "page": 1})
        
        # Mock _get_retrieval_context
        with patch.object(self.handler, '_get_retrieval_context', return_value=([mock_doc], None, "Test Question")):
            # Mock chain.stream
            mock_chain = MagicMock()
            # It yields chunks
            mock_chain.stream.return_value = ["Hello", " World"]
            
            # Setup prompt | llm chain construction
            # In code: chain = self.prompt | self.llm
            # We need to mock the result of this OR operation
            self.handler.prompt.__or__.return_value = mock_chain
            
            # Run streaming
            generator = self.handler.answer_question_stream("Test Question")
            
            # Collect results
            results = list(generator)
            
            # Verify results
            print(f"Generated {len(results)} updates.")
            
            # 1. First chunk
            self.assertEqual(results[0][0], "Hello")
            self.assertEqual(results[0][1], [])
            
            # 2. Second chunk (accumulated)
            self.assertEqual(results[1][0], "Hello World")
            self.assertEqual(results[1][1], [])
            
            # 3. Final result with sources
            final_ans = results[-1][0]
            print(f"Final Answer: {final_ans}")
            self.assertIn("Hello World", final_ans)
            self.assertIn("信息来源", final_ans)
            self.assertIn("test.pdf", final_ans)
            
            # Verify source list in final yield
            final_sources = results[-1][1]
            self.assertEqual(len(final_sources), 1)
            self.assertEqual(final_sources[0]['source'], "test.pdf")

if __name__ == '__main__':
    unittest.main()