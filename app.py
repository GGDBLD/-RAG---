import gradio as gr
import os
import shutil
from src.vector_store import vector_store
from src.qa_chain import qa_chain
from src.acoustic_tools import AcousticCalculator
from src.utils import extract_top_keywords, generate_knowledge_charts

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

def chat_response(message, history, scene_env, sonar_type):
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

def run_calculator(calc_mode, r, f, tl_type, sl, tl, nl, di, ts, active_mode, t, s, d, v_s, v_t, f0, ts_type, ts_r, ts_l, inv_fom, inv_f, inv_type):
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

with gr.Blocks(title="水声工程智能分析系统", theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# 🌊 水声工程智能分析系统 (Intelligent Underwater Acoustic System)")
    
    with gr.Row():
        # === 左侧边栏 (Sidebar) ===
        with gr.Column(scale=1, min_width=300, variant="panel"):
            gr.Markdown("### 🛠️ 专业工具箱")
            
            # 1. 场景设定
            with gr.Group():
                gr.Markdown("#### 1. 场景设定 (Context)")
                scene_env = gr.Dropdown(
                    choices=["通用/默认", "浅海探测 (多途严重)", "深海探测 (汇聚区)", "极地冰下"],
                    value="通用/默认",
                    label="应用环境"
                )
                sonar_type = gr.Radio(
                    choices=["主动声纳", "被动声纳"],
                    value="被动声纳",
                    label="声纳类型"
                )
            
            # 2. 水声计算器
            with gr.Group():
                gr.Markdown("#### 2. 水声计算器 (Calculator)")
                calc_mode = gr.Dropdown(
                    choices=[
                        "传播损失 (TL)", 
                        "声纳方程 (DT/SNR)", 
                        "声速估算 (SSP)", 
                        "多普勒频移 (Doppler)", 
                        "目标强度估算 (TS)",
                        "逆向求解: 最大探测距离 (R_max)"
                    ],
                    value="传播损失 (TL)",
                    label="计算模式"
                )
                
                # TL 参数
                with gr.Column(visible=True) as grp_tl:
                    tl_r = gr.Number(label="距离 R (km)", value=10)
                    tl_f = gr.Number(label="频率 f (kHz)", value=1)
                    tl_type = gr.Dropdown(choices=["spherical", "cylindrical", "hybrid"], value="spherical", label="扩展类型")

                # Sonar Eq 参数
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

                # SSP 参数
                with gr.Column(visible=False) as grp_ssp:
                    ssp_t = gr.Number(label="温度 T (°C)", value=15)
                    ssp_s = gr.Number(label="盐度 S (ppt)", value=35)
                    ssp_d = gr.Number(label="深度 D (m)", value=100)

                # Doppler 参数
                with gr.Column(visible=False) as grp_doppler:
                    dop_vs = gr.Number(label="声源速度 (节)", value=10)
                    dop_vt = gr.Number(label="目标速度 (节)", value=0)
                    dop_f0 = gr.Number(label="中心频率 (Hz)", value=3000)

                # TS 参数
                with gr.Column(visible=False) as grp_ts_calc:
                    ts_type_sel = gr.Dropdown(choices=["sphere", "cylinder", "submarine"], value="sphere", label="目标类型")
                    ts_rad = gr.Number(label="半径 (m)", value=1)
                    ts_len = gr.Number(label="长度 (m)", value=10)

                # 逆向求解参数
                with gr.Column(visible=False) as grp_inv_rmax:
                    inv_fom = gr.Number(label="品质因数 FOM (dB)", value=80, info="允许的最大传播损失")
                    inv_f = gr.Number(label="频率 f (kHz)", value=1)
                    inv_type = gr.Dropdown(choices=["spherical", "cylindrical", "hybrid"], value="spherical", label="扩展类型")

                with gr.Row():
                    calc_btn = gr.Button("🚀 计算", variant="secondary", scale=1)
                    send_chat_btn = gr.Button("📤 发送结果", scale=1)
                
                calc_result = gr.Textbox(label="计算结果", lines=3)

                # 模式切换逻辑
                def change_mode(mode):
                    return {
                        grp_tl: gr.update(visible=(mode=="传播损失 (TL)")),
                        grp_sonar: gr.update(visible=(mode=="声纳方程 (DT/SNR)")),
                        grp_ssp: gr.update(visible=(mode=="声速估算 (SSP)")),
                        grp_doppler: gr.update(visible=(mode=="多普勒频移 (Doppler)")),
                        grp_ts_calc: gr.update(visible=(mode=="目标强度估算 (TS)")),
                        grp_inv_rmax: gr.update(visible=(mode=="逆向求解: 最大探测距离 (R_max)"))
                    }
                
                calc_mode.change(change_mode, calc_mode, [grp_tl, grp_sonar, grp_ssp, grp_doppler, grp_ts_calc, grp_inv_rmax])
                
                # 计算逻辑绑定和发送结果绑定已移动到文件底部，确保组件已定义

            # 3. 知识洞察
            with gr.Group():
                gr.Markdown("#### 3. 知识洞察 (Insight)")
                
            # 3. 知识洞察
            with gr.Group():
                gr.Markdown("#### 3. 知识洞察 (Insight)")
                
                # 统计图表展示区 (替代原词云)
                chart_image = gr.Image(label="知识库热度榜 (Top 5)", visible=False, show_label=True, height=300)
                
                with gr.Accordion("📊 详细数据", open=False):
                    stat_output = gr.JSON(label="知识库规模")
                    refresh_stat_btn = gr.Button("🔄 刷新排行榜")
                
                gr.Markdown("**🔥 热门术语 (点击查看关联问题)**")
                with gr.Column(): # 改用垂直排列
                    # 5个热词按钮 (Top 5) - 使用 primary 样式更醒目
                    hot_btn1 = gr.Button("1. 加载中...", variant="primary")
                    hot_btn2 = gr.Button("2. 加载中...", variant="primary")
                    hot_btn3 = gr.Button("3. 加载中...", variant="primary")
                    hot_btn4 = gr.Button("4. 加载中...", variant="primary")
                    hot_btn5 = gr.Button("5. 加载中...", variant="primary")
                
                # 推荐问题显示区 (初始隐藏，点击热词后显示)
                with gr.Group(visible=False) as q_group:
                    gr.Markdown("##### 💡 推荐问题 (点击发送):")
                    q_btn1 = gr.Button("问题1", size="sm", variant="secondary")
                    q_btn2 = gr.Button("问题2", size="sm", variant="secondary")
                    q_btn3 = gr.Button("问题3", size="sm", variant="secondary")

                def refresh_insight():
                    stats, keywords, chart_path = get_knowledge_stats()
                    
                    # 格式化热词按钮文本
                    btns = []
                    for i in range(5):
                        if i < len(keywords):
                            word, count = keywords[i]
                            # 使用更清晰的格式
                            btns.append(f"{i+1}. {word} ({count})")
                        else:
                            btns.append("-")
                            
                    # 更新图表
                    chart_update = gr.update(value=chart_path, visible=True) if chart_path else gr.update(visible=False)
                    
                    return stats, chart_update, btns[0], btns[1], btns[2], btns[3], btns[4]

                refresh_stat_btn.click(
                    refresh_insight,
                    inputs=[],
                    outputs=[stat_output, chart_image, hot_btn1, hot_btn2, hot_btn3, hot_btn4, hot_btn5]
                )
                
                # 点击热词 -> 更新推荐问题 -> 显示问题区
                def on_hot_word_click(btn_text):
                    # 从按钮文本提取关键词 "1. 传播损失 (50)" -> "传播损失"
                    if not btn_text or btn_text == "-":
                        return gr.update(visible=False), "Q1", "Q2", "Q3"
                    
                    # 修正正则提取中文词
                    # 格式可能是 "1. 传播损失 (50)" 或 "传播损失"
                    # 尝试更健壮的匹配：取第一个非数字非点非空格的连续字符串
                    # 或者简单 split
                    try:
                        # 假设格式是 "排名. 词 (频次)"
                        parts = btn_text.split('.')
                        if len(parts) > 1:
                            # 取第二部分 " 传播损失 (50)"
                            middle = parts[1].strip()
                            # 去掉最后的 (数字)
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
                
                # 绑定热词点击事件
                hot_btn1.click(on_hot_word_click, hot_btn1, [q_group, q_btn1, q_btn2, q_btn3])
                hot_btn2.click(on_hot_word_click, hot_btn2, [q_group, q_btn1, q_btn2, q_btn3])
                hot_btn3.click(on_hot_word_click, hot_btn3, [q_group, q_btn1, q_btn2, q_btn3])
                hot_btn4.click(on_hot_word_click, hot_btn4, [q_group, q_btn1, q_btn2, q_btn3])
                hot_btn5.click(on_hot_word_click, hot_btn5, [q_group, q_btn1, q_btn2, q_btn3])

                # 绑定推荐问题点击 -> 发送
                def click_question(q):
                    if q and q != "无更多推荐":
                        return q
                    return gr.update()

                # q_btn bindings moved to end of file to ensure msg is defined

        # === 右侧主界面 (Main Panel) ===
        with gr.Column(scale=3):
            gr.Markdown("### 🤖 智能问答与分析")
            
            chatbot = gr.Chatbot(
                label="对话记录", 
                height=600
                # type="messages" # Removed for compatibility with older Gradio versions
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="请输入问题",
                    placeholder="例如：在浅海环境下，多途效应对探测距离有什么影响？",
                    lines=2,
                    scale=4
                )
                send = gr.Button("发送", variant="primary", scale=1)
            
            with gr.Row():
                clear = gr.Button("🧹 清空对话")
                upload_btn_modal = gr.Button("📂 管理知识库文档")

            # 示例问题
            gr.Markdown("#### 💡 猜你想问")
            with gr.Row():
                gr.Button("如何计算传播损失？").click(lambda: "如何计算传播损失？", None, msg)
                gr.Button("浅海声传播有什么特点？").click(lambda: "浅海声传播有什么特点？", None, msg)
                gr.Button("被动声纳方程是什么？").click(lambda: "被动声纳方程是什么？", None, msg)

            # 知识库管理弹窗 (用 Accordion 模拟，Gradio 暂无原生 Modal)
            kb_accordion = gr.Accordion("📚 知识库管理 (点击展开)", open=False)
            with kb_accordion:
                with gr.Tab("上传文档"):
                    f_in = gr.File(label="上传文件")
                    t_in = gr.Radio(["core", "supplement"], value="core", label="类型")
                    u_btn = gr.Button("上传并入库")
                    u_out = gr.Textbox(label="结果")
                    u_btn.click(upload_and_process, [f_in, t_in], u_out)
                with gr.Tab("同步 Data 目录"):
                    s_btn = gr.Button("扫描并同步")
                    s_out = gr.Textbox(label="结果")
                    s_btn.click(sync_data_folder_ui, None, s_out)

            # 绑定按钮点击事件，切换 Accordion 的可见性或打开状态
            # 注意：Gradio 的 Accordion 没有直接的 open 属性可以动态绑定，通常用 visible 或 render
            # 这里我们简单实现：点击按钮切换 Accordion 的 open 状态 (需要 Gradio 4.x+)
            # 或者更简单的：按钮只是一个提示，让用户去点下面的 Accordion
            # 为了更好的体验，我们尝试用 render visibility
            
            def toggle_kb_panel():
                return gr.update(open=True)
            
            upload_btn_modal.click(toggle_kb_panel, None, kb_accordion)

    # ================= 事件绑定 (Event Bindings) =================
    # 必须放在所有组件定义之后，防止 NameError

    # 1. 聊天事件
    send.click(chat_response, [msg, chatbot, scene_env, sonar_type], [msg, chatbot])
    msg.submit(chat_response, [msg, chatbot, scene_env, sonar_type], [msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False)

    # 2. 计算器事件
    calc_btn.click(
        run_calculator,
        inputs=[calc_mode, tl_r, tl_f, tl_type, eq_sl, eq_tl, eq_nl, eq_di, eq_ts, eq_active, ssp_t, ssp_s, ssp_d, dop_vs, dop_vt, dop_f0, ts_type_sel, ts_rad, ts_len, inv_fom, inv_f, inv_type],
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
    
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
