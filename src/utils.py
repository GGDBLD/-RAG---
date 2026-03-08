import logging
import os
import sys
from typing import List, Counter, Tuple
import jieba
import re
import matplotlib.pyplot as plt
import io

def setup_logger(name: str, log_file: str = 'app.log', level=logging.INFO) -> logging.Logger:
    # ... (same as before)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File Handler
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to setup file handler for logger: {e}")
            
    return logger

def extract_top_keywords(text_list: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
    """
    提取文本列表中的高频专业热词及其频次
    :param text_list: 文档内容列表
    :param top_n: 返回前N个热词
    :return: [(词, 频次), ...]
    """
    if not text_list:
        return [("暂无数据", 0)]
        
    combined_text = " ".join(text_list)
    
    # 1. 增加自定义水声词典
    acoustic_terms = [
        "传播损失", "声纳方程", "多途效应", "混响", "声源级", "噪声级", "指向性指数",
        "检测阈", "声速剖面", "汇聚区", "深海声道", "浅海", "波束形成", "匹配滤波",
        "Wenz曲线", "空化噪声", "目标强度", "多普勒", "水听器", "换能器",
        "主动声纳", "被动声纳", "信噪比", "虚警概率", "阵列增益"
    ]
    for term in acoustic_terms:
        jieba.add_word(term)
        
    # 2. 分词
    words = jieba.cut(combined_text)
    
    # 3. 过滤停用词和短词
    filtered_words = []
    for w in words:
        if len(w) >= 2 and not re.match(r'^\d+$', w): # 长度>=2且非纯数字
            filtered_words.append(w)
            
    # 4. 统计词频
    word_counts = Counter(filtered_words)
    
    # 5. 优先筛选专业术语，并按频次排序
    # 先把所有专业术语提出来
    term_counts = []
    for term in acoustic_terms:
        if word_counts[term] > 0:
            term_counts.append((term, word_counts[term]))
            
    # 如果专业术语不够，再补其他高频词
    if len(term_counts) < top_n:
        top_general = word_counts.most_common(top_n * 2)
        for w, c in top_general:
            # 避免重复添加
            if w not in [t[0] for t in term_counts]:
                term_counts.append((w, c))
            if len(term_counts) >= top_n:
                break
    
    # 按频次降序排列
    term_counts.sort(key=lambda x: x[1], reverse=True)
            
    return term_counts[:top_n]

def generate_knowledge_charts(top_keywords: List[Tuple[str, int]]) -> str:
    """
    生成 Top 5 热词排行榜（横向柱状图）
    :param top_keywords: [(词, 频次), ...]
    """
    if not top_keywords:
        return None
        
    # 取前5个
    top5 = top_keywords[:5]
    # 倒序排列，让第一名在最上面（Matplotlib barh 是从下往上画）
    top5.reverse()
    
    words = [t[0] for t in top5]
    counts = [t[1] for t in top5]
    
    # 设置中文字体
    font_path = "msyh.ttc"
    # Windows 字体路径
    if os.path.exists(r"C:\Windows\Fonts\msyh.ttc"):
        font_path = r"C:\Windows\Fonts\msyh.ttc"
    elif os.path.exists(r"C:\Windows\Fonts\simhei.ttf"):
        font_path = r"C:\Windows\Fonts\simhei.ttf"
    
    from matplotlib.font_manager import FontProperties
    font_prop = FontProperties(fname=font_path)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(words, counts, color='skyblue')
    
    # 设置标题和标签
    ax.set_title("Top 5 Knowledge Ranking (High Frequency Terms)", fontsize=12)
    ax.set_xlabel("Frequency Count")
    
    # 设置y轴标签字体（支持中文）
    ax.set_yticklabels(words, fontproperties=font_prop)
    
    # 在柱子上标数值
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center')
                
    # 去掉右边和上边的边框
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图片
    chart_path = "temp_ranking_chart.png"
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close(fig)
    
    return chart_path
