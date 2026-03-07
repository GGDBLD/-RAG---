import os
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rapidocr_onnxruntime import RapidOCR

def create_test_image(text="OCR测试：水声工程与深度学习结合", filename="test_ocr.png"):
    """创建一个包含中文和英文的测试图片"""
    width, height = 800, 200
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # 尝试加载中文字体，如果没有则使用默认字体（可能不支持中文）
    try:
        font = ImageFont.truetype("msyh.ttc", 40) # 微软雅黑
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
            print("Warning: Could not load specific font, using default.")

    # 绘制文字
    draw.text((50, 80), text, fill='black', font=font)
    
    # 保存图片
    image.save(filename)
    print(f"✅ Created test image: {filename}")
    return filename

def test_rapidocr():
    print("-" * 50)
    print("🚀 Starting RapidOCR Verification Test")
    print("-" * 50)

    # 1. 初始化模型
    print("1. Initializing RapidOCR engine...")
    try:
        start_time = time.time()
        # 实例化 RapidOCR，使用默认模型
        engine = RapidOCR()
        init_time = time.time() - start_time
        print(f"✅ RapidOCR initialized successfully in {init_time:.4f}s")
    except Exception as e:
        print(f"❌ Failed to initialize RapidOCR: {e}")
        return

    # 2. 准备测试图片
    img_path = create_test_image()

    # 3. 执行识别
    print(f"\n2. Running OCR on {img_path}...")
    try:
        start_time = time.time()
        # rapidocr 接受图片路径、numpy array 或 bytes
        result, elapse = engine(img_path)
        ocr_time = time.time() - start_time
        
        print(f"✅ OCR completed in {ocr_time:.4f}s")
        print(f"   Internal processing time: {elapse}s")
        
        if result:
            print("\n3. OCR Results:")
            for i, res in enumerate(result):
                # res format: [box_points, text, confidence]
                text = res[1]
                confidence = res[2]
                print(f"   Line {i+1}: '{text}' (Confidence: {confidence:.4f})")
            
            # 简单验证
            if "水声工程" in result[0][1]:
                print("\n✅ Verification PASSED: Detected expected text.")
            else:
                print("\n⚠️ Verification WARNING: Expected text not fully detected.")
        else:
            print("\n❌ OCR returned no results.")

    except Exception as e:
        print(f"❌ OCR execution failed: {e}")

    # 清理测试文件
    try:
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"\n🧹 Cleaned up test file: {img_path}")
    except:
        pass

if __name__ == "__main__":
    test_rapidocr()
