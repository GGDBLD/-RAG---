
import unittest
import sys
import os
import math

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.acoustic_tools import AcousticCalculator
from src.utils import extract_top_keywords, generate_knowledge_charts
from src.qa_chain import QAChainHandler

class TestAcousticCalculator(unittest.TestCase):
    
    def test_transmission_loss_spherical(self):
        # 20 * log10(1000m) = 20 * 3 = 60 dB
        result = AcousticCalculator.calc_transmission_loss(r_km=1.0, f_khz=0, type="spherical")
        self.assertIn("60.00 dB", result)
        self.assertIn("球面扩展", result)

    def test_transmission_loss_cylindrical(self):
        # 10 * log10(1000m) = 10 * 3 = 30 dB
        result = AcousticCalculator.calc_transmission_loss(r_km=1.0, f_khz=0, type="cylindrical")
        self.assertIn("30.00 dB", result)
        self.assertIn("柱面扩展", result)

    def test_transmission_loss_absorption(self):
        # With f=10kHz, alpha should be > 0
        result = AcousticCalculator.calc_transmission_loss(r_km=10.0, f_khz=10.0, type="spherical")
        self.assertIn("吸收损耗", result)

    def test_sonar_equation_active(self):
        # Active: SL - 2TL + TS - (NL - DI) = SE
        # 201 - 2*80 + 10 - (70 - 20) = 201 - 160 + 10 - 50 = 1
        result = AcousticCalculator.calc_sonar_equation(sl=201, tl=80, nl=70, di=20, ts=10, is_active=True)
        self.assertIn("(SNR) = 1.00 dB", result)
        self.assertIn("可能被探测到", result)

    def test_sonar_equation_passive(self):
        # Passive: SL - TL - (NL - DI) = SE
        # 150 - 80 - (70 - 20) = 70 - 50 = 20
        result = AcousticCalculator.calc_sonar_equation(sl=150, tl=80, nl=70, di=20, is_active=False)
        self.assertIn("(SNR) = 20.00 dB", result)
        self.assertIn("SL - TL", result)

    def test_sound_speed(self):
        # Standard sea water approx 1500 m/s
        result = AcousticCalculator.estimate_sound_speed(temp_c=15, sal_ppt=35, depth_m=0)
        self.assertIn("1506", result) # Mackenzie approx value

    def test_doppler_shift(self):
        # Source moving towards target (positive relative velocity)
        # 10 knots approx 5 m/s. 
        # f0 = 1000 Hz. c = 1500.
        # delta_f = 2 * v / c * f0 = 2 * 5.14 / 1500 * 1000 = 10.28 / 1.5 = 6.8 Hz
        result = AcousticCalculator.calc_doppler_shift(v_source_knots=10, v_target_knots=0, f0_hz=1000)
        self.assertIn("多普勒频移", result)
        # Check if positive
        self.assertFalse("-" in result.split("Hz")[0].split("≈")[1]) 

    def test_target_strength_sphere(self):
        # Sphere R=2m. TS = 20log(2) - 6 = 20*0.3 - 6 = 6 - 6 = 0 dB (approx, simplified formula used in code might differ slightly)
        # Code: 10 * log10(R^2 / 4) = 10 * log10(1) = 0
        result = AcousticCalculator.estimate_target_strength(type="sphere", radius_m=2.0)
        self.assertIn("0.0 dB", result)

    def test_inverse_solver(self):
        # FOM = 60 dB. Spherical (20logR). f=0.
        # 20logR = 60 => logR = 3 => R = 1000m = 1km.
        result = AcousticCalculator.solve_max_range(fom=60, f_khz=0, type="spherical")
        self.assertIn("1.00 km", result)

class TestUtils(unittest.TestCase):

    def test_extract_top_keywords(self):
        text = ["声纳方程是水声工程的核心。", "传播损失决定了探测距离。", "声纳方程描述了信噪比。", "多途效应影响通信。"]
        # '声纳方程' appears 2 times. '传播损失' 1 time.
        keywords = extract_top_keywords(text, top_n=5)
        # Check if format is list of tuples
        self.assertTrue(isinstance(keywords, list))
        # Check if '声纳方程' is in the list
        words = [k[0] for k in keywords]
        self.assertIn("声纳方程", words)
        
    def test_generate_chart(self):
        data = [("声纳", 10), ("噪声", 8), ("混响", 5)]
        path = generate_knowledge_charts(data)
        self.assertTrue(os.path.exists(path))
        # Cleanup
        if os.path.exists(path):
            os.remove(path)

class TestQAChainPrompt(unittest.TestCase):

    def test_prompt_includes_calc_mode(self):
        handler = QAChainHandler()
        self.assertIn("【计算模式】", handler.prompt.template)
        rendered = handler.prompt.format(
            context="片段1: 示例内容",
            question="计算多普勒频移，声源速度10节，目标速度5节，中心频率2000Hz",
            env_context="通用/默认",
            device_context="主动声纳"
        )
        self.assertIn("【核心公式】", rendered)
        self.assertIn("【代入计算】", rendered)

if __name__ == '__main__':
    print("🌊 正在运行水声系统自动化测试套件...")
    unittest.main()
