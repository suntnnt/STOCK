import streamlit as st
import requests
import concurrent.futures
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import re

# ==========================================
# 0. 页面配置与 UI 样式 (深色专业版)
# ==========================================

st.set_page_config(
    page_title="股票多智能体分析系统",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# 注入 CSS：深色渐变 + 科技线条 + 同花顺风格
st.markdown("""
<style>
    /* 1. 全局背景：深色渐变 + 网格纹理 */
    .stApp {
        background-color: #0E1117;
        background-image: 
            linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px),
            radial-gradient(circle at 50% 0%, #1e1e24 0%, #0E1117 80%);
        background-size: 40px 40px, 40px 40px, 100% 100%;
        color: #E0E0E0;
    }
    
    /* 2. 侧边栏：深灰磨砂 */
    [data-testid="stSidebar"] {
        background-color: #161920 !important;
        border-right: 1px solid #2D3748;
    }
    
    /* 3. 卡片样式：深色毛玻璃 (Glassmorphism) */
    .agent-card {
        background: rgba(30, 34, 45, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        height: 360px;
        overflow-y: auto;
        display: flex; flex-direction: column;
        transition: transform 0.2s, border-color 0.2s;
    }
    .agent-card:hover {
        transform: translateY(-2px);
        border-color: #00B4D8;
        box-shadow: 0 8px 15px rgba(0, 180, 216, 0.15);
    }

    /* 滚动条美化 */
    .agent-card::-webkit-scrollbar { width: 4px; }
    .agent-card::-webkit-scrollbar-thumb { background: #4A5568; border-radius: 2px; }
    .agent-card::-webkit-scrollbar-track { background: transparent; }

    /* 卡片头部 */
    .card-header { 
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 12px; padding-bottom: 10px; 
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .agent-info { display: flex; align-items: center; gap: 10px; }
    .avatar {
        width: 42px; height: 42px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid #2D3748;
    }
    .agent-name { font-weight: 700; color: #F0F0F0; font-size: 1em; }
    .agent-role { font-size: 0.75em; color: #94A3B8; font-weight: 500; }
    
    /* AI 模型标签 (醒目) */
    .model-badge { 
        font-size: 0.7em; padding: 3px 8px; border-radius: 4px; 
        font-family: 'JetBrains Mono', monospace; font-weight: bold;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .badge-gemini { background: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-deepseek { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-qwen { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }
    
    /* 内容区域 */
    .card-content { 
        font-size: 14px; line-height: 1.6; color: #CBD5E1; 
        white-space: pre-wrap;
    }
    
    /* 按钮优化：霓虹蓝 */
    .stButton>button { 
        background: linear-gradient(90deg, #0077B6, #00B4D8);
        color: white; border: none; 
        font-weight: 600; border-radius: 8px; height: 45px; 
        box-shadow: 0 0 10px rgba(0, 180, 216, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        transform: scale(1.02); 
        box-shadow: 0 0 20px rgba(0, 180, 216, 0.6);
    }
    
    /* 输入框样式 */
    .stTextInput>div>div>input {
        background-color: #1A202C;
        color: white;
        border: 1px solid #4A5568;
        border-radius: 8px;
    }
    
    /* --- 作者署名 (居中标题下方) --- */
    .author-container {
        text-align: center;
        margin-top: -15px;
        margin-bottom: 30px;
    }
    .author-tag {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 4px 16px; 
        border-radius: 20px;
        color: #94A3B8; 
        font-size: 13px; 
        font-weight: 500;
        font-family: "Microsoft YaHei", sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 核心配置 (Agents)
# ==========================================
AGENTS_CONFIG = {
    "macro_analyst": {
        "name": "宏观政策分析师", 
        "role": "Macro Analyst",
        "avatar": "https://randomuser.me/api/portraits/men/32.jpg",
        "provider": "Gemini", 
        "prompt": "你是资深A股宏观政策分析师。输出风格：客观、前瞻。\n任务：结合当前A股环境判断宏观水位。\n输出Markdown列表(200字内)：\n- **宏观评级**：[宽松/中性/紧缩]\n- **核心结论**：(一句话狠话)\n- **政策风口**：(简述)"
    },
    "industry_expert": {
        "name": "行业轮动专家", 
        "role": "Industry Expert",
        "avatar": "https://randomuser.me/api/portraits/women/44.jpg",
        "provider": "Gemini",
        "prompt": "你是A股行业轮动专家。输出风格：突出资金偏好。\n任务：分析当前最强主线。\n输出Markdown列表(150字内)：\n- **最强主线**：(前三名)\n- **轮动预判**：(资金下一步去哪)"
    },
    "funds_analyst": {
        "name": "资金流向分析师", 
        "role": "Funds Analyst",
        "avatar": "https://randomuser.me/api/portraits/men/85.jpg",
        "provider": "Gemini",
        "prompt": "你是资金流向专家。输出风格：看穿对手盘。\n任务：分析五档盘口挂单，判断主力意图。\n输出Markdown列表(200字内)：\n- **资金意图**：[吸筹/吸盘/出货/观望]\n- **盘口密码**：(买一卖一挂单解读)\n- **短线合力**：[强/弱]"
    },
    "technical_analyst": {
        "name": "技术分析专家", 
        "role": "Technical Analyst",
        "avatar": "https://randomuser.me/api/portraits/men/22.jpg",
        "provider": "DeepSeek",
        "prompt": "你是机构技术分析专家。输出风格：点位优先。\n任务：基于开盘/现价/五档盘口，判断趋势。\n输出Markdown列表(200字内)：\n- **技术形态**：[多头/空头/震荡]\n- **买卖区间**：买入[价格]/卖出[价格]/止损[价格]\n- **胜率预估**：[数字]%"
    },
    "fundamental_analyst": {
        "name": "基本面估值分析师", 
        "role": "Value Analyst",
        "avatar": "https://randomuser.me/api/portraits/women/68.jpg",
        "provider": "DeepSeek",
        "prompt": "你是价值投资专家。\n任务：判断估值水位。\n输出Markdown列表(150字内)：\n- **估值水位**：[低估/合理/泡沫]\n- **核心逻辑**：(一句话)"
    },
    "manager_fundamental": {
        "name": "基本面研究总监", 
        "role": "Research Director",
        "avatar": "https://randomuser.me/api/portraits/men/50.jpg",
        "provider": "DeepSeek",
        "prompt": "你是基本面总监。任务：整合报告，做出裁决。\n输出Markdown列表(200字内)：\n- **基本面总评**：[S/A/B/C/D]级\n- **核心矛盾**：(最大利好或利空)\n- **中期趋势**：[看涨/看平/看跌]"
    },
    "manager_momentum": {
        "name": "市场动能总监", 
        "role": "Momentum Director",
        "avatar": "https://randomuser.me/api/portraits/men/46.jpg",
        "provider": "DeepSeek",
        "prompt": "你是动能总监。任务：整合技术和资金面。\n输出Markdown列表(200字内)：\n- **动能状态**：[爆发/跟随/衰竭/死水]\n- **爆发概率**：[数字]%\n- **关键信号**：(最缺什么或最强什么)"
    },
    "risk_system": {
        "name": "系统性风险总监", 
        "role": "Risk Director",
        "avatar": "https://randomuser.me/api/portraits/men/90.jpg",
        "provider": "Qwen", 
        "prompt": "你是系统风险总监。风格：偏执理性。\n任务：找出所有可能崩盘的原因。\n输出Markdown列表(200字内)：\n- **崩盘风险**：[低/中/高]\n- **最大回撤预警**：(最坏情况)"
    },
    "risk_portfolio": {
        "name": "组合风险总监", 
        "role": "Portfolio Risk",
        "avatar": "https://randomuser.me/api/portraits/women/33.jpg",
        "provider": "DeepSeek",
        "prompt": "你是风控精算师。\n任务：给出具体风控指标。\n输出Markdown列表(200字内)：\n- **建议仓位**：[数字]%\n- **止损间距**：[数字]%\n- **流动性预警**：(成交量建议)"
    },
    "general_manager": {
        "name": "投资决策总经理 (GM)", 
        "role": "General Manager",
        "avatar": "https://randomuser.me/api/portraits/men/1.jpg",
        "provider": "DeepSeek",
        "prompt": """你是拥有唯一决策权的GM。风格：狼性、激进但克制。
综合前9位专家报告。

**核心任务：** 根据用户的【持仓成本】和【当前浮动盈亏】，给出具体的操作建议。
- 如果用户亏损：分析是否应该补仓摊低成本（T+0），还是割肉止损？
- 如果用户盈利：分析是否应该止盈离场，还是继续持有？

【输出结构】
### 📊 多空一致性
(强多/偏多/中性/偏空/强空)
### 💡 持仓操作建议 (必填)
(针对用户的持仓成本，给出如“在61.0附近补仓做T”、“现价止盈”等具体建议)
### 🧭 最终指令
【🟢 买入 / 🟡 观望 / 🔴 卖出】
### 📌 建议仓位
【0-100%】
### 📈 实战点位
- **买入区间：** [价格]
- **卖出区间：** [价格]
### 🛑 止损红线
- **价格：** [单一数字]
"""
    }
}

# ==========================================
# 2. 数据服务
# ==========================================

def search_stock_realtime(keyword):
    """实时搜索"""
    url = f"http://suggest3.sinajs.cn/suggest/type=&key={keyword}&name=suggestdata_{int(datetime.now().timestamp())}"
    try:
        res = requests.get(url, headers={'Referer': 'https://finance.sina.com.cn/'})
        content = res.content.decode('gbk', 'ignore')
        if '=""' in content: return None, None
        data_str = content.split('="')[1].split('";')[0]
        parts = data_str.split(',')
        if len(parts) > 5: return parts[5], parts[4]
        return None, None
    except: return None, None

def get_realtime_data_tencent(symbol):
    """腾讯财经接口"""
    code = symbol.lower()
    if not (code.startswith('sh') or code.startswith('sz')):
        if code.startswith('6'): code = f"sh{code}"
        elif code.startswith('0') or code.startswith('3'): code = f"sz{code}"
    
    url = f"http://qt.gtimg.cn/q={code}"
    try:
        res = requests.get(url, timeout=5)
        content = res.content.decode('gbk', 'ignore')
        if 'v_pv_none' in content or len(content) < 20: return None, "无数据"
        data = content.split('="')[1].split('";')[0].split('~')
        if len(data) < 30: return None, "数据异常"
        
        return {
            'name': data[1], 'code': data[2], 'now': float(data[3]),
            'yestend': float(data[4]), 'open': float(data[5]),
            'volume': float(data[6]), 
            'sell1_p': data[19], 'sell1_v': data[20],
            'sell2_p': data[21], 'sell2_v': data[22],
            'sell3_p': data[23], 'sell3_v': data[24],
            'sell4_p': data[25], 'sell4_v': data[26],
            'sell5_p': data[27], 'sell5_v': data[28],
            'buy1_p': data[9],   'buy1_v': data[10],
            'buy2_p': data[11],  'buy2_v': data[12],
            'buy3_p': data[13],  'buy3_v': data[14],
            'buy4_p': data[15],  'buy4_v': data[16],
            'buy5_p': data[17],  'buy5_v': data[18],
            'high': float(data[33]), 'low': float(data[34]),
            'amount': float(data[37]) * 10000,
        }, None
    except Exception as e: return None, str(e)

def get_kline_data_eastmoney(symbol):
    try:
        clean_code = re.sub(r"[^0-9]", "", symbol)
        market = "1" if symbol.startswith("sh") or clean_code.startswith("6") else "0"
        secid = f"{market}.{clean_code}"
        url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {"secid": secid, "fields1": "f1,f2,f3,f4,f5,f6", "fields2": "f51,f52,f53,f54,f55,f57", "klt": "101", "fqt": "1", "end": "20500101", "lmt": "120"}
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
        if data and data.get("data") and data["data"].get("klines"):
            klines = data["data"]["klines"]
            parsed = [{"Date": k.split(',')[0], "Open": float(k.split(',')[1]), "Close": float(k.split(',')[2]), "High": float(k.split(',')[3]), "Low": float(k.split(',')[4]), "Volume": float(k.split(',')[5])} for k in klines]
            return pd.DataFrame(parsed)
        return None
    except: return None

def get_min_data_eastmoney(symbol):
    try:
        clean_code = re.sub(r"[^0-9]", "", symbol)
        market = "1" if symbol.startswith("sh") or clean_code.startswith("6") else "0"
        secid = f"{market}.{clean_code}"
        url = "http://push2his.eastmoney.com/api/qt/stock/trends2/get"
        params = {"secid": secid, "fields1": "f1,f2,f3,f4,f5,f6,f7,f8", "fields2": "f51,f53,f58"}
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
        if data and data.get("data") and data["data"].get("trends"):
            trends = data["data"]["trends"]
            parsed = []
            for t in trends:
                s = t.split(',')
                parsed.append({"Time": s[0].split(' ')[1] if ' ' in s[0] else s[0], "Price": float(s[1]), "Vol": float(s[2])})
            return pd.DataFrame(parsed)
        return None
    except: return None

def call_ai_api(prompt, system_prompt, provider, api_keys, gemini_model_name="gemini-2.5-flash"):
    try:
        if provider == "Gemini":
            if not api_keys.get('gemini'): return "⚠️ 缺 Gemini Key"
            import google.generativeai as genai
            genai.configure(api_key=api_keys['gemini'])
            try:
                model = genai.GenerativeModel(gemini_model_name)
                response = model.generate_content(f"【系统指令】\n{system_prompt}\n\n【用户任务】\n{prompt}")
                return response.text
            except Exception as e:
                return f"Gemini Error: {str(e)}"
        elif provider == "DeepSeek":
            if not api_keys.get('deepseek'): return "⚠️ 缺 DeepSeek Key"
            from openai import OpenAI
            client = OpenAI(api_key=api_keys['deepseek'], base_url="https://api.deepseek.com")
            resp = client.chat.completions.create(model="deepseek-chat", messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}])
            return resp.choices[0].message.content
        elif provider == "Qwen":
            if not api_keys.get('qwen'): return "⚠️ 缺 Qwen Key"
            from openai import OpenAI
            client = OpenAI(api_key=api_keys['qwen'], base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
            resp = client.chat.completions.create(model="qwen-plus", messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}])
            return resp.choices[0].message.content
    except Exception as e: return f"[{provider} Error] {str(e)}"

# ==========================================
# 3. 主界面逻辑
# ==========================================

# 1. 标题区（大标题 + 作者署名）
st.markdown("<h1 style='text-align: center; color: #E2E8F0; font-size: 2.8em; margin-bottom: 0; text-shadow: 0 0 20px rgba(0,180,216,0.3);'>股票多智能体分析系统</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="author-container">
    <div class="author-tag">
        <span>👨‍💻</span>
        <span>作者：红桥小胖侠</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. 侧边栏
with st.sidebar:
    st.title("⚙️ 系统控制")
    secret_gemini = st.secrets.get("GEMINI_API_KEY", "")
    secret_deepseek = st.secrets.get("DEEPSEEK_API_KEY", "")
    secret_qwen = st.secrets.get("QWEN_API_KEY", "")

    with st.expander("🔑 API Key 设置", expanded=True):
        st.caption("优先使用云端 Secrets，此处留空即可。")
        user_gemini = st.text_input("Gemini Key", type="password")
        user_deepseek = st.text_input("DeepSeek Key", type="password")
        user_qwen = st.text_input("Qwen Key", type="password")

        gemini_key = user_gemini if user_gemini else secret_gemini
        deepseek_key = user_deepseek if user_deepseek else secret_deepseek
        qwen_key = user_qwen if user_qwen else secret_qwen
        
        if gemini_key: st.caption("✅ Gemini Ready")
        if deepseek_key: st.caption("✅ DeepSeek Ready")
        if qwen_key: st.caption("✅ Qwen Ready")
    
    st.markdown("---")
    st.subheader("🧠 模型调度")
    gemini_model = st.radio("Gemini 版本:", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-pro"], index=0)
    mode = st.radio("分析策略:", ["混合模式 (推荐)", "全 DeepSeek"], index=0)
    
    st.markdown("---")
    st.subheader("💼 持仓信息")
    has_pos = st.checkbox("我持有此股票", value=True)
    if has_pos:
        cost_price = st.number_input("持仓成本", value=62.08, step=0.1, format="%.2f")
        hold_vol = st.number_input("持仓数量", value=1200, step=100)
    else:
        cost_price = 0.0
        hold_vol = 0

if 'analysis_results' not in st.session_state: st.session_state.analysis_results = {}
if 'market_context' not in st.session_state: st.session_state.market_context = None

# 3. 搜索区
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    user_input = st.text_input("输入股票", value="600276", placeholder="代码 / 名称 / 拼音", label_visibility="collapsed")
    start_btn = st.button("🚀 启动分析委员会", use_container_width=True)

if start_btn:
    api_key_set = {'gemini': gemini_key, 'deepseek': deepseek_key, 'qwen': qwen_key}
    
    with st.status("🔍 正在搜索股票...", expanded=True) as status:
        search_code = user_input.strip()
        if re.match(r'^\d{6}$', search_code): real_symbol, stock_name = search_code, "查询中..."
        else: real_symbol, stock_name = search_stock_realtime(search_code)
        
        if not real_symbol:
            if re.match(r'^[a-zA-Z]{2}\d{6}$', search_code): real_symbol, stock_name = search_code, "直接代码"
            else: status.update(label="❌ 未找到股票", state="error"); st.error("未找到股票"); st.stop()
            
        status.update(label=f"锁定标的: {stock_name} ({real_symbol})", state="running")
        stock_data, err = get_realtime_data_tencent(real_symbol)
        if err: status.update(label="❌ 数据获取失败", state="error"); st.error(f"Error: {err}"); st.stop()
        
        kline_df = get_kline_data_eastmoney(real_symbol)
        min_df = get_min_data_eastmoney(real_symbol)
        
        # 头部行情数据
        change_amt = stock_data['now'] - stock_data['yestend']
        change_pct = (change_amt / stock_data['yestend'] * 100) if stock_data['yestend'] else 0
        color_val = "#FF3B30" if change_amt > 0 else "#00F0F0" # 同花顺红绿风格
        
        st.session_state.market_context = stock_data
        
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f"<div style='text-align:center; font-size:24px; font-weight:bold; color:{color_val}'>¥{stock_data['now']:.2f}<br><span style='font-size:16px'>{change_pct:+.2f}%</span></div>", unsafe_allow_html=True)
        k2.metric("成交量", f"{stock_data['volume']/10000:.0f}万手")
        k3.metric("最高", f"¥{stock_data['high']:.2f}")
        k4.metric("最低", f"¥{stock_data['low']:.2f}")
        
        # --- 图表绘制 (模仿同花顺深色风格) ---
        tab1, tab2 = st.tabs(["📉 分时图 (实时)", "📊 K线图 (日线)"])
        
        chart_layout_common = dict(
            plot_bgcolor='#111111', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#888'),
            xaxis=dict(showgrid=True, gridcolor='#333', zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='#333', zeroline=False),
            margin=dict(l=0, r=0, t=10, b=0)
        )

        with tab1: 
            if min_df is not None and not min_df.empty:
                yestend = stock_data['yestend']
                max_diff = max(abs(min_df['Price'].max() - yestend), abs(min_df['Price'].min() - yestend))
                if max_diff == 0: max_diff = yestend * 0.01
                y_range = [yestend - max_diff * 1.1, yestend + max_diff * 1.1]

                fig_min = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                # 分时线 (黄色/白色)
                fig_min.add_trace(go.Scatter(x=min_df['Time'], y=min_df['Price'], mode='lines', name='价格', line=dict(color='#FFFFFF', width=1.5), fill='tozeroy', fillcolor='rgba(255, 255, 255, 0.1)'), row=1, col=1)
                fig_min.add_hline(y=yestend, line_dash="dash", line_color="#FF0000", line_width=1, row=1, col=1)
                
                # 成交量 (红涨绿跌)
                colors = ['#FF3B30' if row['Price'] >= (min_df.iloc[i-1]['Price'] if i>0 else yestend) else '#00F0F0' for i, row in min_df.iterrows()]
                fig_min.add_trace(go.Bar(x=min_df['Time'], y=min_df['Vol'], name='成交量', marker_color=colors), row=2, col=1)

                fig_min.update_layout(height=400, **chart_layout_common)
                fig_min.update_yaxes(range=y_range, tickformat=".2f", row=1, col=1)
                fig_min.update_yaxes(showticklabels=False, row=2, col=1)
                fig_min.update_xaxes(showticklabels=False, row=1, col=1)
                st.plotly_chart(fig_min, use_container_width=True)
            else: st.info("分时数据暂不可用")
            
        with tab2:
            if kline_df is not None:
                fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                # K线 (红涨绿跌)
                fig_k.add_trace(go.Candlestick(
                    x=kline_df['Date'], open=kline_df['Open'], high=kline_df['High'], low=kline_df['Low'], close=kline_df['Close'],
                    increasing_line_color='#FF3B30', decreasing_line_color='#00F0F0',
                    increasing_fillcolor='#FF3B30', decreasing_fillcolor='#00F0F0'
                ), row=1, col=1)
                
                # 成交量
                colors_k = ['#FF3B30' if row['Close'] >= row['Open'] else '#00F0F0' for i, row in kline_df.iterrows()]
                fig_k.add_trace(go.Bar(x=kline_df['Date'], y=kline_df['Volume'], marker_color=colors_k), row=2, col=1)
                
                fig_k.update_layout(height=400, xaxis_rangeslider_visible=False, showlegend=False, **chart_layout_common)
                fig_k.update_xaxes(showticklabels=False, row=1, col=1)
                fig_k.update_yaxes(showticklabels=False, row=2, col=1)
                st.plotly_chart(fig_k, use_container_width=True)
            else: st.info("K线数据暂不可用")

        # Context Prep
        holding_info = "用户无持仓。"
        if has_pos and cost_price > 0 and hold_vol > 0:
            profit = (stock_data['now'] - cost_price) * hold_vol
            profit_pct = (stock_data['now'] - cost_price) / cost_price * 100
            holding_info = f"用户持仓: 成本 {cost_price}，股数 {hold_vol}，盈亏 {profit:.2f} ({profit_pct:.2f}%)"
        
        market_context = f"股票: {stock_data['name']}({real_symbol}) 现价: {stock_data['now']} 涨跌: {change_pct:.2f}% {holding_info}"
        status.update(label="✅ 数据准备就绪，开始分析", state="complete")

    # AI Execution
    def run_agent(agent_key):
        cfg = AGENTS_CONFIG[agent_key]
        target_provider = cfg["provider"] if "混合" in mode else "DeepSeek"
        res = call_ai_api(market_context, cfg["prompt"], target_provider, api_key_set, gemini_model)
        return agent_key, res, target_provider

    st.session_state.analysis_results = {}
    with st.spinner("🚀 AI 委员会正在分析 (Gemini/DeepSeek 并行中)..."):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_agent, key) for key in list(AGENTS_CONFIG.keys())[:5]]
            for f in concurrent.futures.as_completed(futures):
                k, r, p = f.result()
                st.session_state.analysis_results[k] = {"text": r, "provider": p}
                
    stage1_text = "\n".join([f"{AGENTS_CONFIG[k]['name']}: {v['text']}" for k, v in st.session_state.analysis_results.items()])
    
    with st.spinner("🔄 总监正在整合策略..."):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for k in ["manager_fundamental", "manager_momentum"]:
                cfg = AGENTS_CONFIG[k]
                target_provider = cfg["provider"] if "混合" in mode else "DeepSeek"
                futures.append(executor.submit(call_ai_api, f"行情:{market_context}\n报告:{stage1_text}", cfg["prompt"], target_provider, api_key_set, gemini_model))
            res = [f.result() for f in futures]
            st.session_state.analysis_results["manager_fundamental"] = {"text": res[0], "provider": "DeepSeek"}
            st.session_state.analysis_results["manager_momentum"] = {"text": res[1], "provider": "DeepSeek"}

    stage2_text = stage1_text + "\n" + res[0] + "\n" + res[1]
    
    with st.spinner("🛡️ 风控系统正在计算 (Qwen 介入)..."):
         with concurrent.futures.ThreadPoolExecutor() as executor:
             futures = []
             for k in ["risk_system", "risk_portfolio"]:
                cfg = AGENTS_CONFIG[k]
                # 逻辑修正：Risk System 强制用 Qwen
                target_provider = cfg["provider"] if "混合" in mode else "DeepSeek"
                if k == "risk_system" and "混合" in mode: target_provider = "Qwen"
                
                futures.append(executor.submit(call_ai_api, f"市场:{stage2_text}", cfg["prompt"], target_provider, api_key_set, gemini_model))
             res = [f.result() for f in futures]
             st.session_state.analysis_results["risk_system"] = {"text": res[0], "provider": "Qwen" if "混合" in mode else "DeepSeek"}
             st.session_state.analysis_results["risk_portfolio"] = {"text": res[1], "provider": "DeepSeek"}

    final_text = stage2_text + "\n" + res[0] + "\n" + res[1]
    with st.spinner("👑 总经理最终决策..."):
        k = "general_manager"
        cfg = AGENTS_CONFIG[k]
        target_provider = cfg["provider"] if "混合" in mode else "DeepSeek"
        res = call_ai_api(f"所有报告:\n{final_text}", cfg["prompt"], target_provider, api_key_set, gemini_model)
        st.session_state.analysis_results[k] = {"text": res, "provider": target_provider}
    
    st.success("分析完成！")

# 4. 渲染卡片
def render_section(title, agent_keys, cols=1):
    st.subheader(title)
    columns = st.columns(cols)
    for i, key in enumerate(agent_keys):
        cfg = AGENTS_CONFIG[key]
        result_obj = st.session_state.analysis_results.get(key)
        content = result_obj["text"] if result_obj else "等待指令..."
        provider = result_obj["provider"] if result_obj else "OFFLINE"
        if provider == "Gemini": provider = gemini_model.split("-")[0]
        
        # 标签颜色类
        badge_class = "badge-gemini"
        if "DeepSeek" in provider: badge_class = "badge-deepseek"
        if "Qwen" in provider: badge_class = "badge-qwen"

        with columns[i % cols]:
            st.markdown(f"""
            <div class="agent-card">
                <div class="card-header">
                    <div class="agent-info">
                        <img src="{cfg['avatar']}" class="avatar">
                        <div>
                            <div class="agent-name">{cfg['name']}</div>
                            <div class="agent-role">{cfg['role']}</div>
                        </div>
                    </div>
                    <span class="model-badge {badge_class}">{provider}</span>
                </div>
                <div class="card-content">{content}</div>
            </div>
            """, unsafe_allow_html=True)

render_section("🔍 第一阶段：多维分析 (Gemini/DeepSeek)", list(AGENTS_CONFIG.keys())[:5], cols=5)
render_section("🧠 第二阶段：策略博弈 (DeepSeek)", ["manager_fundamental", "manager_momentum"], cols=2)
render_section("🛡️ 第三阶段：风控委员会 (Qwen/DeepSeek)", ["risk_system", "risk_portfolio"], cols=2)

gm_res = st.session_state.analysis_results.get("general_manager")
if gm_res:
    st.markdown("---")
    st.subheader("🏆 最终决议")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e1e24 0%, #2d1b2e 100%); border: 1px solid #FF3B30; border-radius: 18px; padding: 30px; box-shadow: 0 0 30px rgba(255, 59, 48, 0.2);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; border-bottom:1px solid rgba(255, 255, 255, 0.1); padding-bottom:15px;">
            <div style="display:flex; align-items:center; gap:15px;">
                <img src="{AGENTS_CONFIG['general_manager']['avatar']}" style="width:60px; height:60px; border-radius:50%; border:2px solid #FF3B30;">
                <div>
                    <span style="font-size:1.5em; font-weight:bold; color:#FFFFFF;">👑 投资决策总经理</span>
                    <div style="color:#A0A0A0; font-size:0.9em;">General Manager</div>
                </div>
            </div>
            <span class="model-badge badge-deepseek">DeepSeek V3</span>
        </div>
        <div style="font-size:1.1em; line-height:1.8; color:#E0E0E0; white-space: pre-wrap;">{gm_res['text']}</div>
    </div>
    """, unsafe_allow_html=True)