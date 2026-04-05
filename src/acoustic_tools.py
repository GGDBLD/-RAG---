import math
import numpy as np

class AcousticCalculator:
    """
    水声工程专用计算工具库
    包含传播损失、声速、声纳方程等核心公式
    """
    
    @staticmethod
    def calc_transmission_loss(r_km: float, f_khz: float = 0, type: str = "spherical") -> str:
        """
        计算传播损失 (Transmission Loss, TL)
        :param r_km: 距离 (km)
        :param f_khz: 频率 (kHz), 用于计算吸收系数 (如果为0则忽略吸收)
        :param type: 扩展类型 ('spherical', 'cylindrical', 'practical')
        :return: 格式化的结果字符串
        """
        if r_km <= 0:
            return "错误: 距离必须大于0"
            
        r_m = r_km * 1000.0
        
        # 1. 几何扩展损失
        if type == "spherical":
            tl_geom = 20 * math.log10(r_m)
            desc = "球面扩展 (20logR)"
        elif type == "cylindrical":
            tl_geom = 10 * math.log10(r_m)
            desc = "柱面扩展 (10logR)"
        else: # practical / hybrid
            # 简化的工程经验公式: 近场球面，远场柱面 + 过渡
            # 这里使用经典的 Marsh-Schulkin 模型简化版: 
            # TL = 20logR (r < H), TL = 10logR + 10logH (r > H)
            # 暂按混合模型: 15logR
            tl_geom = 15 * math.log10(r_m)
            desc = "混合扩展 (15logR)"
            
        # 2. 吸收损失 (Thorpe公式)
        tl_abs = 0.0
        alpha = 0.0
        if f_khz > 0:
            f2 = f_khz ** 2
            # Thorpe公式 (dB/km)
            alpha = (0.11 * f2 / (1 + f2)) + (44 * f2 / (4100 + f2)) + (2.75e-4 * f2) + 0.003
            tl_abs = alpha * r_km
            
        total_tl = tl_geom + tl_abs
        
        result = f"传播损失 TL = {total_tl:.2f} dB\n"
        result += f"- 几何扩展: {tl_geom:.2f} dB ({desc})\n"
        if f_khz > 0:
            result += f"- 吸收损耗: {tl_abs:.2f} dB (α ≈ {alpha:.4f} dB/km @ {f_khz}kHz)"
            
        return result

    @staticmethod
    def calc_sonar_equation(sl: float, tl: float, nl: float, di: float, ts: float = 0, is_active: bool = False) -> str:
        """
        声纳方程计算 (Sonar Equation)
        :param sl: 声源级 (dB)
        :param tl: 单程传播损失 (dB)
        :param nl: 噪声级 (dB)
        :param di: 指向性指数 (dB)
        :param ts: 目标强度 (dB), 仅主动声纳需要
        :param is_active: 是否为主动声纳
        :return: 剩余信噪比 (Signal Excess) 和 检测阈建议
        """
        # 信号部分 (Signal)
        if is_active:
            # 主动: SL - 2TL + TS
            signal = sl - 2 * tl + ts
            eq_str = "SL - 2TL + TS"
        else:
            # 被动: SL - TL
            signal = sl - tl
            eq_str = "SL - TL"
            
        # 噪声部分 (Noise)
        noise = nl - di
        
        # 剩余信噪比 (Signal Excess / SNR at receiver input)
        se = signal - noise
        
        result = f"接收端信噪比 (SNR) = {se:.2f} dB\n"
        result += f"计算公式: ({eq_str}) - (NL - DI)\n"
        result += f"- 信号项: {signal:.2f} dB\n"
        result += f"- 噪声项: {noise:.2f} dB\n"
        
        if se > 0:
            result += ">> 目标 **可能被探测到** (SNR > 0 dB)"
        elif se > -5:
            result += ">> 目标 **探测困难** (SNR 在临界区)"
        else:
            result += ">> 目标 **无法探测** (SNR < -5 dB)"
            
        return result

    @staticmethod
    def estimate_sound_speed(temp_c: float, sal_ppt: float, depth_m: float) -> str:
        """
        声速估算 (Mackenzie 公式简化版)
        :param temp_c: 温度 (摄氏度)
        :param sal_ppt: 盐度 (ppt, 千分比)
        :param depth_m: 深度 (m)
        """
        # Mackenzie (1981) 九项公式简化版
        c = 1448.96 + 4.591 * temp_c - 5.304e-2 * temp_c**2 + 2.374e-4 * temp_c**3 \
            + 1.340 * (sal_ppt - 35) + 1.630e-2 * depth_m + 1.675e-7 * depth_m**2 \
            - 1.025e-2 * temp_c * (sal_ppt - 35) - 7.139e-13 * temp_c * depth_m**3
            
        return f"估算声速 c ≈ {c:.2f} m/s\n(基于 Mackenzie 经验公式: T={temp_c}°C, S={sal_ppt}‰, D={depth_m}m)"

    @staticmethod
    def calc_doppler_shift(v_source_knots: float, v_target_knots: float, f0_hz: float, c: float = 1500.0) -> str:
        """
        多普勒频移计算 (Doppler Shift)
        :param v_source_knots: 声源速度 (节, 1节=0.5144m/s, 正值接近)
        :param v_target_knots: 目标速度 (节, 正值接近)
        :param f0_hz: 发射中心频率 (Hz)
        :param c: 声速 (m/s)
        """
        # 1节 = 0.51444 m/s
        KNOT_TO_MS = 0.51444
        v_s = v_source_knots * KNOT_TO_MS
        v_t = v_target_knots * KNOT_TO_MS
        
        # 相对速度 (接近为正)
        # 简化公式：Δf = f0 * (v_relative / c)
        # 完整公式 (收发合置): f_recv = f0 * (c + v_t) / (c - v_s)
        # 这里假设是主动声纳回波模型 (双程多普勒)
        # v_relative = v_s + v_t (相向运动速度之和)
        
        # 双程多普勒 (主动声纳): Δf ≈ 2 * v_radial / c * f0
        v_radial = v_s + v_t
        delta_f = 2 * v_radial / c * f0_hz
        
        result = f"多普勒频移 Δf ≈ {delta_f:.2f} Hz\n"
        result += f"- 相对径向速度: {v_radial/KNOT_TO_MS:.1f} 节 ({v_radial:.1f} m/s)\n"
        result += f"- 频移比例: {(delta_f/f0_hz)*100:.4f}%\n"
        result += "(注：假设为主动声纳双程多普勒，正值表示目标接近)"
        return result

    @staticmethod
    def estimate_target_strength(type: str, radius_m: float = 0, length_m: float = 0) -> str:
        """
        目标强度估算 (Target Strength, TS)
        :param type: 目标类型 ('sphere', 'cylinder', 'submarine')
        :param radius_m: 半径 (m)
        :param length_m: 长度 (m), 仅圆柱体/潜艇需要
        """
        ts = -999.0
        desc = ""
        
        if type == "sphere":
            # 刚性球体 TS = 10log(R^2/4) = 20logR - 6
            if radius_m <= 0: return "错误: 半径必须大于0"
            ts = 10 * math.log10((radius_m**2) / 4)
            desc = f"刚性球体 (R={radius_m}m)"
            
        elif type == "cylinder":
            # 刚性圆柱体 (正横方向) TS = 10log(R*L^2 / (2*lambda)) -> 需频率，这里用简化的高频近似
            # 近似 TS = 10log(0.5 * R * L) (几何近似，不严谨但工程常用估算)
            if radius_m <= 0 or length_m <= 0: return "错误: 尺寸必须大于0"
            ts = 10 * math.log10(0.5 * radius_m * length_m) 
            desc = f"圆柱体几何近似 (R={radius_m}m, L={length_m}m)"
            
        elif type == "submarine":
            # 二战潜艇经验值: TS = 10log(0.18 * L * W) -> W假设为D
            # 或者简单经验值
            if length_m <= 0: return "错误: 长度必须大于0"
            # 经验公式 TS = 10log(L^2) - 20 (正横)
            ts = 10 * math.log10(length_m**2) - 20
            desc = f"潜艇正横经验值 (L={length_m}m)"
            
        return f"估算目标强度 TS ≈ {ts:.1f} dB\n- 模型: {desc}"

    @staticmethod
    def solve_max_range(fom: float, f_khz: float = 1.0, type: str = "spherical") -> str:
        """
        逆向求解：计算最大探测距离 (Max Range)
        已知 FOM (Figure of Merit) = TL_max
        求解方程: TL(r) = FOM
        其中 TL(r) = GeomLoss(r) + AbsLoss(r) = K*log10(r) + alpha*r*1e-3
        由于包含 r 和 log(r)，需使用数值解法（二分法）。
        
        :param fom: 品质因数 (允许的最大传播损失, dB)
        :param f_khz: 频率 (kHz)
        :param type: 扩展类型
        """
        if fom <= 0:
            return "错误: FOM 必须大于 0"
            
        # 1. 确定 K 值
        if type == "spherical": k = 20
        elif type == "cylindrical": k = 10
        else: k = 15
        
        # 2. 计算吸收系数 alpha (dB/km)
        alpha = 0.0
        if f_khz > 0:
            f2 = f_khz ** 2
            alpha = (0.11 * f2 / (1 + f2)) + (44 * f2 / (4100 + f2)) + (2.75e-4 * f2) + 0.003
            
        # 3. 定义目标函数: f(r) = TL(r) - FOM
        # r 单位为 km
        def tl_func(r_km):
            if r_km <= 1e-6: return -fom
            # TL = k * log10(r_m) + alpha * r_km
            # 注意：几何扩展通常基于米，但吸收基于公里。需统一单位。
            # 标准公式 TL = k*log10(r_m) + alpha*r_km
            # r_m = r_km * 1000
            return (k * math.log10(r_km * 1000)) + (alpha * r_km) - fom

        # 4. 二分查找求解
        low = 0.001 # 1m
        high = 10000.0 # 10000km (足够大)
        
        # 先检查边界
        if tl_func(low) > 0: return "最大距离 < 1m (FOM太小)"
        if tl_func(high) < 0: return "最大距离 > 10000km (FOM极大)"
        
        # 迭代 50 次 (精度足够)
        for _ in range(50):
            mid = (low + high) / 2
            if tl_func(mid) < 0:
                low = mid
            else:
                high = mid
                
        r_result = (low + high) / 2
        
        result = f"最大探测距离 R_max ≈ {r_result:.2f} km\n"
        result += f"- 依据: FOM = TL = {fom} dB\n"
        result += f"- 模型: {k}logR + {alpha:.4f}R (频率 {f_khz}kHz)"
        return result

    @staticmethod
    def estimate_ambient_noise(sea_state: int, f_khz: float, shipping_traffic: int = 2) -> str:
        """
        估算海洋环境噪声级 (简化版 Wenz 曲线)
        :param sea_state: 海况等级 (0-6)
        :param f_khz: 频率 (kHz)
        :param shipping_traffic: 航运密度 (1: 低, 2: 中, 3: 高)
        """
        f_hz = f_khz * 1000.0
        if f_hz <= 0:
            return "错误: 频率必须大于0"
            
        nl_total = 0.0
        
        # 1. 湍流噪声 (Turbulence) - 低频 < 10Hz
        nl_turb = 106 - 30 * math.log10(f_hz) if f_hz > 0 else 0
        
        # 2. 航运噪声 (Shipping) - 10Hz 到 1000Hz
        traffic_base = {1: 60, 2: 70, 3: 80}.get(shipping_traffic, 70)
        nl_ship = traffic_base - 20 * math.log10(f_hz / 100) if f_hz > 0 else 0
        
        # 3. 风浪噪声 (Wind/Wave) - 100Hz 到 100kHz
        ss_base = 50 + 7.5 * math.sqrt(sea_state)
        nl_wind = ss_base - 17 * math.log10(f_hz / 1000) if f_hz > 0 else 0
        
        # 4. 热噪声 (Thermal) - > 100kHz
        nl_thermal = -15 + 20 * math.log10(f_hz) if f_hz > 0 else 0
        
        # 根据 Wenz 曲线简化区间截断
        intensities = [
            10**(nl_turb/10) if 1 <= f_hz < 100 else 0,
            10**(nl_ship/10) if 10 <= f_hz < 10000 else 0,
            10**(nl_wind/10) if 100 <= f_hz < 100000 else 0,
            10**(nl_thermal/10) if f_hz >= 50000 else 0
        ]
        
        total_intensity = sum(intensities)
        if total_intensity > 0:
            nl_total = 10 * math.log10(total_intensity)
        else:
            nl_total = 0.0

        result = f"估算环境噪声级 NL ≈ {nl_total:.1f} dB (re 1μPa²/Hz)\n"
        result += f"- 参数: 频率 {f_khz}kHz, 海况 {sea_state}, 航运密度等级 {shipping_traffic}\n"
        
        # 推断主要噪声源
        result += "- 主要噪声源: "
        if 10 < f_hz < 500 and nl_ship > nl_wind: 
            result += "航运噪声为主"
        elif 500 <= f_hz < 50000: 
            result += "风浪噪声为主"
        elif f_hz >= 50000: 
            result += "分子热运动为主"
        else: 
            result += "低频湍流或综合背景"
            
        return result

    @staticmethod
    def calc_array_directivity(array_type: str, num_elements: int, spacing_lambda: float) -> str:
        """
        估算阵列指向性指数(DI)与波束宽度
        :param array_type: 阵列类型 ('line', 'planar')
        :param num_elements: 阵元总数
        :param spacing_lambda: 阵元间距(以波长为单位，如 0.5)
        """
        if num_elements <= 0 or spacing_lambda <= 0:
            return "错误: 阵元数和间距必须大于0"
            
        di = 0.0
        beamwidth = 0.0
        desc = ""
        
        if array_type == "line":
            # 直线阵 (侧射) 近似: DI = 10log(2 * N * d/lambda)
            di = 10 * math.log10(2 * num_elements * spacing_lambda)
            # 波束宽度 (度) ≈ 50.8 / (N * d/lambda)
            beamwidth = 50.8 / (num_elements * spacing_lambda)
            desc = "均匀直线阵 (侧射)"
        elif array_type == "planar":
            # 平面阵 (假设方形): DI = 10log( 4pi A / lambda^2 ) ≈ 10log( 4pi N d^2/lambda^2 )
            di = 10 * math.log10(4 * math.pi * num_elements * (spacing_lambda**2))
            # 波束宽度近似 (按边长)
            side_elements = math.sqrt(num_elements)
            beamwidth = 50.8 / (side_elements * spacing_lambda)
            desc = "均匀方形平面阵 (侧射)"
        else:
            return "未知阵列类型"
            
        result = f"阵列性能估算:\n"
        result += f"- 指向性指数 DI ≈ {di:.1f} dB\n"
        result += f"- 主瓣宽度 (3dB) ≈ {beamwidth:.1f}°\n"
        result += f"- 阵列模型: {desc} (阵元数 N={int(num_elements)}, 间距 d={spacing_lambda}λ)"
        return result

