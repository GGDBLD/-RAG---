import gradio as gr
import os
import shutil
from src.vector_store import vector_store
from src.qa_chain import qa_chain
from src.acoustic_tools import AcousticCalculator
from src.utils import extract_top_keywords, generate_knowledge_charts, generate_tl_range_plot

# ================= 辅助函数 =================

def upload_and_process(file_obj, doc_type):
    if not file_obj:
        return "请选择文件。"
    
    try:
        # Create temp directory
        temp_dir = os.path.join(os.getcwd(), "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)

        # Determine source path
        if hasattr(file_obj, 'name'):
            source_path = file_obj.name
        else:
            source_path = str(file_obj)
        
        filename = os.path.basename(source_path)
        saved_path = os.path.join(temp_dir, filename)
        
        shutil.copy2(source_path, saved_path)
        
        if not os.path.exists(saved_path) or os.path.getsize(saved_path) == 0:
            return "上传失败: 文件为空或无法读取。"

        success, msg, num_chunks = vector_store.add_document(saved_path, doc_type)
        
        try:
            if os.path.exists(saved_path):
                os.remove(saved_path)
        except:
            pass
        
        if success:
            return f"成功！文件名: {filename}\n类型: {doc_type}\n新增片段数: {num_chunks}"
        else:
            return f"失败: {msg}"
            
    except Exception as e:
        return f"处理异常: {str(e)}"

def sync_data_folder_ui():
    folder_path = "data"
    if not os.path.exists(folder_path):
        return f"文件夹 {folder_path} 不存在。"
    added_files = vector_store.scan_and_ingest(folder_path)
    if added_files:
        return f"同步成功！已自动添加 {len(added_files)} 个新文件：\n" + "\n".join(added_files)
    else:
        return "data 文件夹中没有发现新文件。"

def get_knowledge_stats():
    """
    获取知识库统计数据和热词 (全量统计)
    """
    try:
        files = vector_store.get_indexed_files()
        
        # 获取向量总数
        total_vectors = 0
        collection = vector_store.vectordb.get()
        if collection and "ids" in collection:
            total_vectors = len(collection["ids"])
            
        stats = {
            "total_files": len(files),
            "total_vectors": total_vectors
        }
        
        # 2. 提取热词 (全量统计)
        keywords = [("暂无数据", 0)]
        chart_path = None
        
        if collection and "documents" in collection and collection["documents"]:
            # 全量统计：虽然有点慢，但能保证准确和一致
            # 如果文档真的超级多（>10万），再考虑采样。现在 1.2万还可以接受。
            all_docs = collection["documents"]
            keywords = extract_top_keywords(all_docs, top_n=5)
            
            # 3. 生成统计图表 (Top 5 柱状图)
            chart_path = generate_knowledge_charts(keywords)
            
        return stats, keywords, chart_path
    except Exception as e:
        return {"total_files": 0, "total_vectors": 0}, [("错误", 0)], None

# 预设的专业术语及关联问题库 (模拟大模型生成的推荐问题)
# 实际项目中可以调用 LLM 生成，但为了响应速度，这里先用预置字典
KEYWORD_QUESTIONS = {
    "传播损失": [
        "什么是传播损失？请解释其物理意义。",
        "浅海和深海的传播损失有什么区别？",
        "如何利用传播损失估算探测距离？"
    ],
    "声纳方程": [
        "主动声纳方程和被动声纳方程的区别是什么？",
        "如何通过声纳方程计算检测阈(DT)？",
        "提高声纳系统增益的主要手段有哪些？"
    ],
    "多途效应": [
        "多途效应对声纳信号处理有什么影响？",
        "如何利用多途结构进行被动测距？",
        "浅海信道中的多途时延扩展大约是多少？"
    ],
    "混响": [
        "什么是体积混响、海面混响和海底混响？",
        "主动声纳如何抑制混响干扰？",
        "多普勒频移能否用于抗混响？"
    ],
    "声源级": [
        "常见舰船的辐射噪声源级大概是多少？",
        "如何测量水下目标的声源级？",
        "声源级与航速有什么关系？"
    ],
    "噪声级": [
        "Wenz曲线描述了什么规律？",
        "海洋环境噪声的主要来源有哪些？",
        "如何降低本舰噪声对声纳的影响？"
    ],
    "指向性指数": [
        "什么是阵列的指向性指数(DI)？",
        "波束宽度与指向性指数有什么关系？",
        "如何设计高指向性的声纳基阵？"
    ],
    "检测阈": [
        "检测阈(DT)与虚警概率有什么关系？",
        "ROC曲线在声纳检测中如何应用？",
        "信噪比低时如何改善检测性能？"
    ],
    "声速剖面": [
        "典型的深海声速剖面结构是怎样的？",
        "什么是汇聚区？它与声速剖面有什么关系？",
        "表面声道是如何形成的？"
    ],
    "多普勒": [
        "水声中的多普勒效应有什么特点？",
        "如何利用多普勒频移估算目标速度？",
        "宽带信号对多普勒敏感吗？"
    ],
    "暂无数据": ["请先上传文档以生成热词", "如何上传文档？", "知识库支持哪些格式？"]
}

def get_related_questions(keyword):
    """根据关键词获取推荐问题"""
    return KEYWORD_QUESTIONS.get(keyword, [f"什么是{keyword}？", f"{keyword}在水声工程中的应用？", f"如何计算{keyword}？"])

# ================= 聊天核心逻辑 =================

def chat_response(message, history, scene_env, sonar_type, sea_state, bottom_type, ssp_type, freq_band, task_goal, array_type):
    """
    处理用户提问，结合侧边栏的场景参数
    """
    if not message:
        yield "", history
        return
    
    # 构造场景化 Prompt
    # 这里的技巧是把场景信息隐式地加到 query 里，或者传给 qa_chain 处理
    # 为了简单起见，我们直接修改 message 的语义，或者让 qa_chain 内部处理
    # 这里我们采用“系统提示注入”的方式，但在 qa_chain.py 没改之前，
    # 我们先用简单的 "Context Injection"
    
    context_prefix = ""
    if scene_env != "通用/默认":
        context_prefix += f"[当前场景：{scene_env}] "
    if sonar_type:
        context_prefix += f"[设备类型：{sonar_type}] "
    # 扩展注入：海况、海底、声速剖面、频段、任务
    if sea_state:
        context_prefix += f"[海况：{sea_state}] "
    if bottom_type:
        context_prefix += f"[海底：{bottom_type}] "
    if ssp_type:
        context_prefix += f"[声速剖面：{ssp_type}] "
    if freq_band:
        context_prefix += f"[频段：{freq_band}] "
    if task_goal:
        context_prefix += f"[任务：{task_goal}] "
    if array_type:
        context_prefix += f"[阵列：{array_type}] "
        
    # 将场景信息传给 QA Chain (需要修改 qa_chain.py 的接口，或者直接拼接到 question)
    # 暂时拼接到 question，让 LLM 感知
    # 注意：实际展示给用户的 message 不变，只是传给 LLM 的 query 变了
    effective_query = f"{context_prefix}{message}"
    
    # Init history
    if history is None:
        history = []
        
    # Append new user message to UI history
    history.append({"role": "user", "content": message})
    
    # Placeholder for assistant response
    history.append({"role": "assistant", "content": "正在思考..."})
    yield "", history
    
    # Call QA Chain
    # 注意：这里我们把 effective_query 传进去
    full_response = ""
    try:
        for answer, _ in qa_chain.answer_question_stream(effective_query, history[:-2]):
            full_response = answer
            history[-1]["content"] = full_response
            yield "", history
    except Exception as e:
        history[-1]["content"] = f"发生错误: {str(e)}"
        yield "", history

# ================= 计算器逻辑 =================

def run_calculator(calc_mode, r, f, tl_type, sl, tl, nl, di, ts, active_mode, t, s, d, v_s, v_t, f0, ts_type, ts_r, ts_l, inv_fom, inv_f, inv_type, wenz_ss, wenz_f, wenz_traffic, array_type_sel, array_n, array_d):
    """
    根据选择的计算模式调用 AcousticCalculator
    """
    try:
        if calc_mode == "传播损失 (TL)":
            return AcousticCalculator.calc_transmission_loss(float(r), float(f), tl_type)
        elif calc_mode == "声纳方程 (DT/SNR)":
            is_active = (active_mode == "主动")
            return AcousticCalculator.calc_sonar_equation(
                float(sl), float(tl), float(nl), float(di), float(ts), is_active
            )
        elif calc_mode == "声速估算 (SSP)":
            return AcousticCalculator.estimate_sound_speed(float(t), float(s), float(d))
        elif calc_mode == "多普勒频移 (Doppler)":
            return AcousticCalculator.calc_doppler_shift(float(v_s), float(v_t), float(f0))
        elif calc_mode == "目标强度估算 (TS)":
            return AcousticCalculator.estimate_target_strength(ts_type, float(ts_r), float(ts_l))
        elif calc_mode == "逆向求解: 最大探测距离 (R_max)":
            return AcousticCalculator.solve_max_range(float(inv_fom), float(inv_f), inv_type)
        elif calc_mode == "环境噪声估算 (Wenz)":
            return AcousticCalculator.estimate_ambient_noise(int(wenz_ss), float(wenz_f), int(wenz_traffic))
        elif calc_mode == "阵列性能估算 (DI)":
            return AcousticCalculator.calc_array_directivity(array_type_sel, int(array_n), float(array_d))
        else:
            return "未知模式"
    except ValueError:
        return "输入错误：请输入有效的数字。"
    except Exception as e:
        return f"计算错误: {str(e)}"

def send_calc_to_chat(result, history):
    """
    将计算结果一键发送到聊天框
    """
    if not result:
        return "", history
    
    # 构造一条带引导的提问
    prompt = f"根据计算结果：\n{result}\n请分析该结果对水声探测性能的具体影响。"
    
    # 直接作为用户消息发送 (这需要触发 chat_response)
    # 但 Gradio 的按钮点击通常只能更新组件状态
    # 这里我们把 prompt 填入 msg 输入框，让用户自己点发送，或者直接触发提交
    return prompt

# ================= UI 构建 =================

with gr.Blocks(title="水声工程智能问答系统") as demo:
    gr.Markdown("# 🌊 水声工程智能问答系统")

    with gr.Tabs(selected="qa") as top_tabs:
        with gr.Tab("问答系统", id="qa"):
            with gr.Row():
                with gr.Column(scale=1, min_width=300, variant="panel"):
                    gr.Markdown("### 🛠️ 专业工具箱")
                    sidebar_menu = gr.Radio(choices=["场景设定", "水声计算器", "知识洞察"], value="场景设定", label="菜单")
                    def toggle_menu(menu):
                        return {
                            grp_scene: gr.update(visible=(menu=="场景设定")),
                            grp_calc: gr.update(visible=(menu=="水声计算器")),
                            grp_insight: gr.update(visible=(menu=="知识洞察")),
                        }
                    
                    with gr.Group() as grp_scene:
                        gr.Markdown("#### 1. 场景设定 (Context)")
                        scene_env = gr.Dropdown(choices=["通用/默认", "浅海探测 (多途严重)", "深海探测 (汇聚区)", "极地冰下"], value="通用/默认", label="应用环境")
                        sonar_type = gr.Radio(choices=["主动声纳", "被动声纳"], value="被动声纳", label="声纳类型")
                        with gr.Row():
                            sea_state = gr.Dropdown(choices=[str(i) for i in range(0,7)], value="3", label="海况等级 (Sea State)")
                            bottom_type = gr.Dropdown(choices=["泥", "砂", "岩"], value="砂", label="海底类型")
                        with gr.Row():
                            ssp_type = gr.Dropdown(choices=["表面声道", "中层极小", "汇聚区"], value="中层极小", label="声速剖面")
                            freq_band = gr.Dropdown(choices=["低频", "中频", "高频"], value="中频", label="频段")
                        array_type = gr.Dropdown(choices=["线阵", "面阵", "拖曳阵"], value="线阵", label="阵列类型")
                        task_goal = gr.Dropdown(choices=["侦察", "跟踪", "定位", "通信"], value="侦察", label="任务目标")
                        scene_summary = gr.Textbox(label="场景摘要", lines=3, interactive=False)
                        preset = gr.Dropdown(choices=["浅海被动侦察（中频/线阵/SS=3/砂底）", "深海主动搜索（高频/面阵/汇聚区）", "港湾通信（低频/SS=2/泥底）"], value=None, label="场景模板")
                        def apply_preset(p):
                            if p == "浅海被动侦察（中频/线阵/SS=3/砂底）":
                                return {scene_env: gr.update(value="浅海探测 (多途严重)"), sonar_type: gr.update(value="被动声纳"), sea_state: gr.update(value="3"), bottom_type: gr.update(value="砂"), ssp_type: gr.update(value="中层极小"), freq_band: gr.update(value="中频"), array_type: gr.update(value="线阵"), task_goal: gr.update(value="侦察")}
                            if p == "深海主动搜索（高频/面阵/汇聚区）":
                                return {scene_env: gr.update(value="深海探测 (汇聚区)"), sonar_type: gr.update(value="主动声纳"), sea_state: gr.update(value="4"), bottom_type: gr.update(value="岩"), ssp_type: gr.update(value="汇聚区"), freq_band: gr.update(value="高频"), array_type: gr.update(value="面阵"), task_goal: gr.update(value="定位")}
                            if p == "港湾通信（低频/SS=2/泥底）":
                                return {scene_env: gr.update(value="通用/默认"), sonar_type: gr.update(value="被动声纳"), sea_state: gr.update(value="2"), bottom_type: gr.update(value="泥"), ssp_type: gr.update(value="表面声道"), freq_band: gr.update(value="低频"), array_type: gr.update(value="线阵"), task_goal: gr.update(value="通信")}
                            return {}
                        preset.change(apply_preset, preset, [scene_env, sonar_type, sea_state, bottom_type, ssp_type, freq_band, array_type, task_goal])
    
                        def update_scene_summary(scene_env_v, sonar_type_v, sea_state_v, bottom_type_v, ssp_type_v, freq_band_v, array_type_v, task_goal_v):
                            lines = [
                                f"应用环境：{scene_env_v}",
                                f"声纳类型：{sonar_type_v}；阵列：{array_type_v}",
                                f"海况：{sea_state_v}；海底：{bottom_type_v}；声速剖面：{ssp_type_v}；频段：{freq_band_v}；任务：{task_goal_v}"
                            ]
                            return "\n".join(lines)
    
                        for c in [scene_env, sonar_type, sea_state, bottom_type, ssp_type, freq_band, array_type, task_goal]:
                            c.change(update_scene_summary, inputs=[scene_env, sonar_type, sea_state, bottom_type, ssp_type, freq_band, array_type, task_goal], outputs=[scene_summary])
                        preset.change(update_scene_summary, inputs=[scene_env, sonar_type, sea_state, bottom_type, ssp_type, freq_band, array_type, task_goal], outputs=[scene_summary])
                    
                    with gr.Group(visible=False) as grp_calc:
                        gr.Markdown("#### 2. 水声计算器 (Calculator)")
                        calc_mode = gr.Dropdown(choices=["传播损失 (TL)", "声纳方程 (DT/SNR)", "声速估算 (SSP)", "多普勒频移 (Doppler)", "目标强度估算 (TS)", "逆向求解: 最大探测距离 (R_max)", "环境噪声估算 (Wenz)", "阵列性能估算 (DI)"], value="传播损失 (TL)", label="计算模式")
                        with gr.Column(visible=True) as grp_tl:
                            tl_r = gr.Number(label="距离 R (km)", value=10)
                            tl_f = gr.Number(label="频率 f (kHz)", value=1)
                            tl_type = gr.Dropdown(choices=["spherical", "cylindrical", "hybrid"], value="spherical", label="扩展类型")
                            with gr.Accordion("📈 传播损失曲线", open=False):
                                tl_plot_btn = gr.Button("生成 TL-Range 曲线", size="sm")
                                tl_plot_img = gr.Image(label="曲线图", type="filepath", visible=False)
                                def plot_tl(f_val, r_val):
                                    max_r = max(20.0, float(r_val) * 1.5)
                                    path = generate_tl_range_plot(float(f_val), max_r)
                                    return gr.update(value=path, visible=True)
                                tl_plot_btn.click(plot_tl, inputs=[tl_f, tl_r], outputs=[tl_plot_img])
                        with gr.Column(visible=False) as grp_sonar:
                            eq_sl = gr.Number(label="声源级 SL (dB)", value=220)
                            eq_tl = gr.Number(label="传播损失 TL (dB)", value=80)
                            eq_nl = gr.Number(label="噪声级 NL (dB)", value=60)
                            eq_di = gr.Number(label="指向性指数 DI (dB)", value=20)
                            eq_active = gr.Radio(["主动", "被动"], value="主动", label="模式")
                            eq_ts = gr.Number(label="目标强度 TS (dB)", value=10, visible=True)
                            def toggle_ts(mode):
                                return gr.update(visible=(mode=="主动"))
                            eq_active.change(toggle_ts, eq_active, eq_ts)
                        with gr.Column(visible=False) as grp_ssp:
                            ssp_t = gr.Number(label="温度 T (°C)", value=15)
                            ssp_s = gr.Number(label="盐度 S (ppt)", value=35)
                            ssp_d = gr.Number(label="深度 D (m)", value=100)
                        with gr.Column(visible=False) as grp_doppler:
                            dop_vs = gr.Number(label="声源速度 (节)", value=10)
                            dop_vt = gr.Number(label="目标速度 (节)", value=0)
                            dop_f0 = gr.Number(label="中心频率 (Hz)", value=3000)
                        with gr.Column(visible=False) as grp_ts_calc:
                            ts_type_sel = gr.Dropdown(choices=["sphere", "cylinder", "submarine"], value="sphere", label="目标类型")
                            ts_rad = gr.Number(label="半径 (m)", value=1)
                            ts_len = gr.Number(label="长度 (m)", value=10)
                        with gr.Column(visible=False) as grp_inv_rmax:
                            inv_fom = gr.Number(label="品质因数 FOM (dB)", value=80, info="允许的最大传播损失")
                            inv_f = gr.Number(label="频率 f (kHz)", value=1)
                            inv_type = gr.Dropdown(choices=["spherical", "cylindrical", "hybrid"], value="spherical", label="扩展类型")
                        with gr.Column(visible=False) as grp_wenz:
                            wenz_ss = gr.Dropdown(choices=[str(i) for i in range(0,7)], value="3", label="海况等级")
                            wenz_f = gr.Number(label="频率 (kHz)", value=1.0)
                            wenz_traffic = gr.Dropdown(choices=[("低", 1), ("中", 2), ("高", 3)], value=2, label="航运密度")
                        with gr.Column(visible=False) as grp_array:
                            array_type_sel = gr.Dropdown(choices=[("直线阵", "line"), ("平面阵", "planar")], value="line", label="阵列类型")
                            array_n = gr.Number(label="阵元总数 N", value=32)
                            array_d = gr.Number(label="阵元间距 (以波长计)", value=0.5, info="例如 0.5 表示半波长")
                        with gr.Row():
                            calc_btn = gr.Button("🚀 计算", variant="secondary", scale=1)
                            send_chat_btn = gr.Button("📤 发送结果", scale=1)
                        calc_result = gr.Textbox(label="计算结果", lines=3)
                        def change_mode(mode):
                            return {grp_tl: gr.update(visible=(mode=="传播损失 (TL)")), grp_sonar: gr.update(visible=(mode=="声纳方程 (DT/SNR)")), grp_ssp: gr.update(visible=(mode=="声速估算 (SSP)")), grp_doppler: gr.update(visible=(mode=="多普勒频移 (Doppler)")), grp_ts_calc: gr.update(visible=(mode=="目标强度估算 (TS)")), grp_inv_rmax: gr.update(visible=(mode=="逆向求解: 最大探测距离 (R_max)")), grp_wenz: gr.update(visible=(mode=="环境噪声估算 (Wenz)")), grp_array: gr.update(visible=(mode=="阵列性能估算 (DI)"))}
                        calc_mode.change(change_mode, calc_mode, [grp_tl, grp_sonar, grp_ssp, grp_doppler, grp_ts_calc, grp_inv_rmax, grp_wenz, grp_array])
    
                    with gr.Group(visible=False) as grp_insight:
                        gr.Markdown("#### 3. 知识洞察 (Insight)")
                        chart_image = gr.HTML(label="知识库热度榜 (Top 5)", visible=False, show_label=True)
                        with gr.Accordion("📊 详细数据", open=False):
                            stat_output = gr.JSON(label="知识库规模")
                            refresh_stat_btn = gr.Button("🔄 刷新排行榜")
                        gr.Markdown("**🔥 热门术语 (点击查看关联问题)**")
                        with gr.Column():
                            hot_btn1 = gr.Button("1. 加载中...", variant="primary")
                            hot_btn2 = gr.Button("2. 加载中...", variant="primary")
                            hot_btn3 = gr.Button("3. 加载中...", variant="primary")
                            hot_btn4 = gr.Button("4. 加载中...", variant="primary")
                            hot_btn5 = gr.Button("5. 加载中...", variant="primary")
                        with gr.Group(visible=False) as q_group:
                            gr.Markdown("##### 💡 推荐问题 (点击发送):")
                            q_btn1 = gr.Button("问题1", size="sm", variant="secondary")
                            q_btn2 = gr.Button("问题2", size="sm", variant="secondary")
                            q_btn3 = gr.Button("问题3", size="sm", variant="secondary")
                        def refresh_insight():
                            stats, keywords, chart_path = get_knowledge_stats()
                            btns = []
                            for i in range(5):
                                if i < len(keywords):
                                    word, count = keywords[i]
                                    btns.append(f"{i+1}. {word} ({count})")
                                else:
                                    btns.append("-")
                            chart_update = gr.update(visible=False)
                            if chart_path and os.path.exists(chart_path):
                                try:
                                    import base64
                                    with open(chart_path, "rb") as f:
                                        b64 = base64.b64encode(f.read()).decode("ascii")
                                    html = f'<div style="width:100%;text-align:center;"><img src="data:image/png;base64,{b64}" alt="Top 5 Ranking" style="max-width:100%;height:auto;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.15);" /></div>'
                                    chart_update = gr.update(value=html, visible=True)
                                except Exception as _:
                                    chart_update = gr.update(visible=False)
                            return stats, chart_update, btns[0], btns[1], btns[2], btns[3], btns[4]
                        refresh_stat_btn.click(refresh_insight, inputs=[], outputs=[stat_output, chart_image, hot_btn1, hot_btn2, hot_btn3, hot_btn4, hot_btn5])
                        sidebar_menu.change(toggle_menu, inputs=[sidebar_menu], outputs=[grp_scene, grp_calc, grp_insight])
                        def on_hot_word_click(btn_text):
                            if not btn_text or btn_text == "-":
                                return gr.update(visible=False), "Q1", "Q2", "Q3"
                            try:
                                parts = btn_text.split('.')
                                if len(parts) > 1:
                                    middle = parts[1].strip()
                                    word = middle.rsplit(' (', 1)[0]
                                else:
                                    word = btn_text
                            except:
                                word = btn_text
                            if word == "加载中..." or word == "暂无数据":
                                return gr.update(visible=False), "Q1", "Q2", "Q3"
                            questions = get_related_questions(word)
                            qs = questions + ["无更多推荐"] * (3 - len(questions))
                            return gr.update(visible=True), qs[0], qs[1], qs[2]
                        hot_btn1.click(on_hot_word_click, hot_btn1, [q_group, q_btn1, q_btn2, q_btn3])
                        hot_btn2.click(on_hot_word_click, hot_btn2, [q_group, q_btn1, q_btn2, q_btn3])
                        hot_btn3.click(on_hot_word_click, hot_btn3, [q_group, q_btn1, q_btn2, q_btn3])
                        hot_btn4.click(on_hot_word_click, hot_btn4, [q_group, q_btn1, q_btn2, q_btn3])
                        hot_btn5.click(on_hot_word_click, hot_btn5, [q_group, q_btn1, q_btn2, q_btn3])
                        def click_question(q):
                            if q and q != "无更多推荐":
                                return q
                            return gr.update()

                with gr.Column(scale=3):
                    gr.Markdown("### 🤖 智能问答与分析")
                    chatbot = gr.Chatbot(label="对话记录", height=600)
                    with gr.Row():
                        msg = gr.Textbox(label="请输入问题", placeholder="例如：在浅海环境下，多途效应对探测距离有什么影响？", lines=2, scale=4)
                        send = gr.Button("发送", variant="primary", scale=1)
                    with gr.Row():
                        clear = gr.Button("🧹 清空对话")
                        upload_btn_modal = gr.Button("📂 知识库管理")
                    gr.Markdown("#### 💡 猜你想问")
                    with gr.Row():
                        gr.Button("如何计算传播损失？").click(lambda: "如何计算传播损失？", None, msg)
                        gr.Button("浅海声传播有什么特点？").click(lambda: "浅海声传播有什么特点？", None, msg)
                        gr.Button("被动声纳方程是什么？").click(lambda: "被动声纳方程是什么？", None, msg)

        with gr.Tab("知识库管理", id="kb"):
            gr.Markdown("## 📚 知识库管理")
            with gr.Tabs():
                with gr.Tab("上传文档"):
                    kb_f_in = gr.File(label="上传文件")
                    kb_t_in = gr.Radio(["core", "supplement"], value="core", label="类型")
                    kb_u_btn = gr.Button("上传并入库", variant="primary")
                    kb_u_out = gr.Textbox(label="结果")
                    kb_u_btn.click(upload_and_process, [kb_f_in, kb_t_in], kb_u_out)
                with gr.Tab("同步 Data 目录"):
                    kb_s_btn = gr.Button("扫描并同步", variant="primary")
                    kb_s_out = gr.Textbox(label="结果")
                    kb_s_btn.click(sync_data_folder_ui, None, kb_s_out)
                with gr.Tab("统计与热词"):
                    kb_refresh_btn = gr.Button("刷新统计", variant="primary")
                    kb_stat_output = gr.JSON(label="知识库规模")
                    kb_keywords = gr.Dataframe(headers=["术语", "频次"], datatype=["str", "number"], row_count=5, column_count=(2, "fixed"))
                    kb_chart_image = gr.HTML(visible=False)
                    kb_list_btn = gr.Button("列出已入库文件")
                    kb_files_out = gr.Textbox(label="已入库文件", lines=10)

                    def refresh_kb_stats():
                        stats, keywords, chart_path = get_knowledge_stats()
                        df = [[w, c] for w, c in (keywords or [])]
                        chart_update = gr.update(visible=False)
                        if chart_path and os.path.exists(chart_path):
                            try:
                                import base64
                                with open(chart_path, "rb") as f:
                                    b64 = base64.b64encode(f.read()).decode("ascii")
                                html = f'<div style="width:100%;text-align:center;"><img src="data:image/png;base64,{b64}" alt="Top 5 Ranking" style="max-width:100%;height:auto;border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.15);" /></div>'
                                chart_update = gr.update(value=html, visible=True)
                            except Exception as _:
                                chart_update = gr.update(visible=False)
                        return stats, df, chart_update

                    def list_indexed_files():
                        files = vector_store.get_indexed_files()
                        files = sorted(files)
                        return "\n".join(files) if files else "暂无已入库文件。"

                    kb_refresh_btn.click(refresh_kb_stats, inputs=[], outputs=[kb_stat_output, kb_keywords, kb_chart_image])
                    kb_list_btn.click(list_indexed_files, inputs=[], outputs=[kb_files_out])

    def go_to_kb():
        return gr.update(selected="kb")

    upload_btn_modal.click(go_to_kb, inputs=[], outputs=[top_tabs])

    # ================= 事件绑定 (Event Bindings) =================
    # 必须放在所有组件定义之后，防止 NameError

    # 保持“手动刷新”显示图表：不在页面加载或跳转时自动刷新

    # 1. 聊天事件
    send.click(chat_response, [msg, chatbot, scene_env, sonar_type, sea_state, bottom_type, ssp_type, freq_band, task_goal, array_type], [msg, chatbot])
    msg.submit(chat_response, [msg, chatbot, scene_env, sonar_type, sea_state, bottom_type, ssp_type, freq_band, task_goal, array_type], [msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)

    # 2. 计算器事件
    calc_btn.click(
        run_calculator,
        inputs=[calc_mode, tl_r, tl_f, tl_type, eq_sl, eq_tl, eq_nl, eq_di, eq_ts, eq_active, ssp_t, ssp_s, ssp_d, dop_vs, dop_vt, dop_f0, ts_type_sel, ts_rad, ts_len, inv_fom, inv_f, inv_type, wenz_ss, wenz_f, wenz_traffic, array_type_sel, array_n, array_d],
        outputs=calc_result
    )

    # 3. 发送结果到对话框
    send_chat_btn.click(
        send_calc_to_chat,
        inputs=[calc_result, chatbot], # history 实际上没用上，但为了接口统一
        outputs=[msg]
    )

    # 4. 知识洞察事件 (Knowledge Insight Events)
    # hot_btn events are handled in the sidebar block
    
    # q_btn events
    q_btn1.click(click_question, q_btn1, msg)
    q_btn2.click(click_question, q_btn2, msg)
    q_btn3.click(click_question, q_btn3, msg)

if __name__ == "__main__":
    # Auto-sync
    print("Startup: Scanning 'data' folder...")
    vector_store.scan_and_ingest("data")
    
    # Try to launch on 7860, but if occupied, gradio will automatically find another port if we remove server_port constraint
    # Or we can specify a starting port and let it auto-increment, but gradio does this by default if server_port is None.
    demo.queue().launch(server_name="127.0.0.1", theme=gr.themes.Soft(primary_hue="indigo"))
