import streamlit as st
import requests
import concurrent.futures
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import re

# ==========================================
# 0. 页面配置与 UI 样式
# ==========================================

st.set_page_config(
    page_title="股票自动多智能分析系统",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# 注入 CSS：优化后的 UI
st.markdown("""
<style>
    /* 1. 全局背景 */
    .stApp {
        background-color: #F5F5F7;
        color: #1D1D1F;
    }
    
    /* 2. 侧边栏 */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E5E5E5;
    }
    
    /* 3. 卡片样式 */
    .agent-card {
        background: #FFFFFF;
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
        height: 380px;
        overflow-y: auto;
        display: flex; flex-direction: column;
        transition: transform 0.2s;
    }
    .agent-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
        border-color: #0071E3;
    }

    /* 滚动条 */
    .agent-card::-webkit-scrollbar { width: 4px; }
    .agent-card::-webkit-scrollbar-thumb { background: #D1D1D6; border-radius: 2px; }

    /* 卡片头部 */
    .card-header { 
        display: flex; align-items: center; gap: 12px;
        margin-bottom: 16px; padding-bottom: 12px; 
        border-bottom: 1px solid #F2F2F7;
    }
    .avatar {
        width: 48px; height: 48px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid #F5F5F7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .agent-name { font-weight: 700; color: #1D1D1F; font-size: 1.05em; }
    .agent-role { font-size: 0.75em; color: #86868B; font-weight: 500; }
    
    /* 模型标签 */
    .model-badge { 
        font-size: 0.65em; padding: 2px 8px; border-radius: 12px; 
        background: #F2F2F7; color: #86868B; border: 1px solid #E5E5E5;
        font-family: monospace;
    }
    
    /* 内容区域 */
    .card-content { 
        font-size: 15px; line-height: 1.6; color: #424245; 
        white-space: pre-wrap;
    }
    
    /* 按钮优化 */
    .stButton>button { 
        background: #0071E3; color: white; border: none; 
        font-weight: 600; border-radius: 10px; height: 45px; 
        box-shadow: 0 4px 10px rgba(0, 113, 227, 0.3);
    }
    .stButton>button:hover { background: #0077ED; transform: scale(1.01); }
    
    /* --- 作者署名 (修改：跟随页面滚动) --- */
    .author-tag {
        position: absolute; /* 改为 absolute，不再是 fixed */
        top: -60px; /* 调整位置到顶部 */
        right: 10px; 
        z-index: 10;
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid #E5E5E5;
        padding: 4px 12px; 
        border-radius: 20px;
        color: #86868B; 
        font-size: 12px; 
        font-weight: 600;
        font-family: "Microsoft YaHei", sans-serif;
        backdrop-filter: blur(5px);
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
        "provider": "DeepSeek",
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
        
        if len(parts) > 5:
            full_code = parts[5] 
            name = parts[4]      
            return full_code, name
            
        return None, None
    except Exception as e:
        print(f"Search Error: {e}")
        return None, None

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
        data_str = content.split('="')[1].split('";')[0]
        data = data_str.split('~')
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
    except Exception as e:
        return None, str(e)

def get_kline_data_eastmoney(symbol):
    try:
        clean_code = re.sub(r"[^0-9]", "", symbol)
        market = "1" if symbol.startswith("sh") or clean_code.startswith("6") else "0"
        secid = f"{market}.{clean_code}"
        
        url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "secid": secid, "fields1": "f1,f2,f3,f4,f5,f6", "fields2": "f51,f52,f53,f54,f55,f57",
            "klt": "101", "fqt": "1", "end": "20500101", "lmt": "120"
        }
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
        if data and data.get("data") and data["data"].get("klines"):
            klines = data["data"]["klines"]
            parsed = []
            for k in klines:
                s = k.split(',')
                parsed.append({"Date": s[0], "Open": float(s[1]), "Close": float(s[2]), 
                               "High": float(s[3]), "Low": float(s[4]), "Volume": float(s[5])})
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
                time_str = s[0].split(' ')[1] if ' ' in s[0] else s[0]
                parsed.append({
                    "Time": time_str, 
                    "Price": float(s[1]), 
                    "Vol": float(s[2])
                })
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
                if "404" in str(e) or "400" in str(e):
                    try:
                        model = genai.GenerativeModel("gemini-pro")
                        res = model.generate_content(f"{system_prompt}\n{prompt}")
                        return f"[自动降级 gemini-pro] {res.text}"
                    except: return f"Gemini Error: {str(e)}"
                return f"Gemini Error: {str(e)}"
            
        elif provider == "DeepSeek":
            if not api_keys.get('deepseek'): return "⚠️ 缺 DeepSeek Key"
            from openai import OpenAI
            client = OpenAI(api_key=api_keys['deepseek'], base_url="https://api.deepseek.com")
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}],
                temperature=0.1
            )
            return resp.choices[0].message.content

        elif provider == "Qwen":
            if not api_keys.get('qwen'): return "⚠️ 缺 Qwen Key"
            from openai import OpenAI
            client = OpenAI(api_key=api_keys['qwen'], base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
            resp = client.chat.completions.create(
                model="qwen-plus",
                messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]
            )
            return resp.choices[0].message.content
    except Exception as e:
        return f"[{provider} Error] {str(e)}"

# ==========================================
# 3. 主界面逻辑 (Key 安全化 + UI 调整)
# ==========================================

# 独立的作者署名，随页面滚动
st.markdown("""
<div class="author-tag">
    <span>👨‍💻</span>
    <span>作者：红桥小胖侠</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("⚙️ 系统控制")
    
    # --- [关键修改] API Key 安全逻辑 ---
    # 逻辑：优先读取 Secrets，如果 Secrets 有值，输入框留空（保护隐私），如果用户强制输入，则覆盖 Secrets
    
    # 1. 尝试从 Secrets 获取
    secret_gemini = st.secrets.get("GEMINI_API_KEY", "")
    secret_deepseek = st.secrets.get("DEEPSEEK_API_KEY", "")
    secret_qwen = st.secrets.get("QWEN_API_KEY", "")

    with st.expander("🔑 API Key 设置", expanded=True):
        st.caption("提示：若已配置云端 Secrets，此处留空即可。输入框内容优先。")
        
        # 输入框默认不显示 Secret，防止截图泄露
        user_gemini = st.text_input("Gemini Key", type="password", placeholder="留空则使用系统默认 Key")
        user_deepseek = st.text_input("DeepSeek Key", type="password", placeholder="留空则使用系统默认 Key")
        user_qwen = st.text_input("Qwen Key", type="password", placeholder="留空则使用系统默认 Key")

        # 最终使用的 Key：用户输入 > Secret
        gemini_key = user_gemini if user_gemini else secret_gemini
        deepseek_key = user_deepseek if user_deepseek else secret_deepseek
        qwen_key = user_qwen if user_qwen else secret_qwen
        
        # 状态指示灯
        if gemini_key: st.caption("✅ Gemini 已就绪")
        if deepseek_key: st.caption("✅ DeepSeek 已就绪")
    
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
        # [关键修改] 无持仓时给予默认安全值，防止计算报错
        cost_price = 0.0
        hold_vol = 0

st.markdown("<h1 style='text-align: center; color: #0071E3;'>股票自动多智能分析系统</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #86868B; font-size: 14px;'>Institutional Grade Multi-Agent System v10.5</p>", unsafe_allow_html=True)

if 'analysis_results' not in st.session_state: st.session_state.analysis_results = {}
if 'market_context' not in st.session_state: st.session_state.market_context = None

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    user_input = st.text_input("输入股票", value="600276", placeholder="代码(600276) / 名称(恒瑞) / 拼音(HRYY)", label_visibility="collapsed")
    start_btn = st.button("🚀 启动分析委员会", use_container_width=True)

if start_btn:
    api_key_set = {'gemini': gemini_key, 'deepseek': deepseek_key, 'qwen': qwen_key}
    
    with st.status("🔍 正在搜索股票...", expanded=True) as status:
        search_code = user_input.strip()
        
        if re.match(r'^\d{6}$', search_code):
            real_symbol = search_code 
            stock_name = "查询中..."
        else:
            real_symbol, stock_name = search_stock_realtime(search_code)
        
        if not real_symbol:
            if re.match(r'^[a-zA-Z]{2}\d{6}$', search_code):
                real_symbol = search_code
                stock_name = "直接代码"
            else:
                status.update(label="❌ 未找到股票", state="error")
                st.error(f"未找到 '{user_input}' 对应的 A 股代码。请尝试直接输入 6 位代码。")
                st.stop()
            
        status.update(label=f"锁定标的: {stock_name} ({real_symbol})", state="running")

        stock_data, err = get_realtime_data_tencent(real_symbol)
        if err: 
            status.update(label="❌ 数据获取失败", state="error"); st.error(f"无法获取数据: {err}"); st.stop()
        
        kline_df = get_kline_data_eastmoney(real_symbol)
        min_df = get_min_data_eastmoney(real_symbol)
        
        change_amt = stock_data['now'] - stock_data['yestend']
        change_pct = (change_amt / stock_data['yestend'] * 100) if stock_data['yestend'] else 0
        color_delta = "inverse" if change_amt < 0 else "normal"
        
        st.session_state.market_context = stock_data
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("当前价格", f"¥{stock_data['now']:.2f}", f"{change_pct:.2f}%", delta_color=color_delta)
        k2.metric("成交量", f"{stock_data['volume']/10000:.0f}万手")
        k3.metric("最高", f"¥{stock_data['high']:.2f}")
        k4.metric("最低", f"¥{stock_data['low']:.2f}")
        
        tab1, tab2 = st.tabs(["📉 分时图 (实时)", "📊 K线图 (日线)"])
        
        with tab1: 
            if min_df is not None and not min_df.empty:
                yestend = stock_data['yestend']
                max_diff = max(abs(min_df['Price'].max() - yestend), abs(min_df['Price'].min() - yestend))
                if max_diff == 0: max_diff = yestend * 0.01
                y_range = [yestend - max_diff * 1.1, yestend + max_diff * 1.1]

                fig_min = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                fig_min.add_trace(go.Scatter(
                    x=min_df['Time'], y=min_df['Price'], mode='lines', name='价格', 
                    line=dict(color='#0071E3', width=2), fill='tozeroy', fillcolor='rgba(0, 113, 227, 0.1)'
                ), row=1, col=1)
                fig_min.add_hline(y=yestend, line_dash="dash", line_color="#86868B", line_width=1, row=1, col=1)
                
                colors = ['#FF3B30' if row['Price'] >= (min_df.iloc[i-1]['Price'] if i>0 else yestend) else '#34C759' for i, row in min_df.iterrows()]
                fig_min.add_trace(go.Bar(x=min_df['Time'], y=min_df['Vol'], name='成交量', marker_color=colors), row=2, col=1)

                fig_min.update_layout(
                    height=380, margin=dict(l=0, r=0, t=10, b=0), 
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                    hovermode='x unified', showlegend=False, font=dict(color='#86868B'),
                    yaxis=dict(range=y_range, tickformat=".2f", gridcolor='rgba(0,0,0,0.05)')
                )
                fig_min.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(size=10))
                fig_min.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.05)', zeroline=False, row=1, col=1)
                fig_min.update_yaxes(showgrid=False, zeroline=False, row=2, col=1, showticklabels=False)
                st.plotly_chart(fig_min, use_container_width=True)
            else: st.info("分时数据暂不可用")
            
        with tab2:
            if kline_df is not None:
                fig_k = go.Figure(data=[go.Candlestick(
                    x=kline_df['Date'], open=kline_df['Open'], high=kline_df['High'], low=kline_df['Low'], close=kline_df['Close'],
                    increasing_line_color='#FF3B30', decreasing_line_color='#34C759'
                )])
                fig_k.update_layout(xaxis_rangeslider_visible=False, height=380, margin=dict(l=0, r=0, t=10, b=0),
                                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                    xaxis=dict(showgrid=False, tickfont=dict(color='#86868B')), 
                                    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', tickfont=dict(color='#86868B')))
                st.plotly_chart(fig_k, use_container_width=True)
            else: st.info("K线数据暂不可用")

        # --- [关键修改] 盈亏计算的 0 值保护 ---
        holding_info = "用户无持仓。"
        if has_pos:
            # 判断：只有当成本价 > 0 且 股数 > 0 时才计算
            if cost_price > 0 and hold_vol > 0:
                profit = (stock_data['now'] - cost_price) * hold_vol
                profit_pct = (stock_data['now'] - cost_price) / cost_price * 100
                holding_info = f"""
                【重要：用户持仓信息】
                - 持仓成本: {cost_price:.3f} 元
                - 持仓股数: {hold_vol} 股
                - 当前盈亏: {profit:.2f} 元 ({profit_pct:.2f}%)
                - 你的决策必须明确：是建议止损离场、继续持有、还是补仓做T？
                """
            else:
                holding_info = "用户已勾选持仓，但成本或股数为0，请忽略具体的盈亏数值，仅给出一般性操作建议。"
        
        market_context = f"""
        [标的] {stock_data['name']} ({real_symbol})
        [现价] {stock_data['now']:.2f} (涨跌: {change_pct:.2f}%)
        [成交] 量:{stock_data['volume']/100:.0f}手 / 额:{stock_data['amount']/10000:.0f}万
        [五档] 买1:{stock_data['buy1_p']}({stock_data['buy1_v']}) ... 卖1:{stock_data['sell1_p']}({stock_data['sell1_v']})
        {holding_info}
        """
        status.update(label="✅ 数据准备就绪，开始分析", state="complete")

    def run_agent(agent_key):
        cfg = AGENTS_CONFIG[agent_key]
        target_provider = cfg["provider"] if "混合" in mode else "DeepSeek"
        res = call_ai_api(market_context, cfg["prompt"], target_provider, api_key_set, gemini_model)
        return agent_key, res, target_provider

    st.session_state.analysis_results = {}
    
    with st.spinner("第一阶段：5位分析师正在并行分析..."):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_agent, key) for key in list(AGENTS_CONFIG.keys())[:5]]
            for f in concurrent.futures.as_completed(futures):
                k, r, p = f.result()
                st.session_state.analysis_results[k] = {"text": r, "provider": p}
    
    stage1_text = "\n".join([f"【{AGENTS_CONFIG[k]['name']}】: {v['text']}" for k, v in st.session_state.analysis_results.items()])
    with st.spinner("第二阶段：总监正在整合..."):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for k in ["manager_fundamental", "manager_momentum"]:
                cfg = AGENTS_CONFIG[k]
                target_provider = cfg["provider"] if "混合" in mode else "DeepSeek"
                futures.append(executor.submit(call_ai_api, f"行情：\n{market_context}\n下属报告：\n{stage1_text}", cfg["prompt"], target_provider, api_key_set, gemini_model))
            res = [f.result() for f in futures]
            st.session_state.analysis_results["manager_fundamental"] = {"text": res[0], "provider": "DeepSeek"}
            st.session_state.analysis_results["manager_momentum"] = {"text": res[1], "provider": "DeepSeek"}

    stage2_text = stage1_text + "\n" + res[0] + "\n" + res[1]
    with st.spinner("第三阶段：风控正在计算..."):
         with concurrent.futures.ThreadPoolExecutor() as executor:
             futures = []
             for k in ["risk_system", "risk_portfolio"]:
                cfg = AGENTS_CONFIG[k]
                target_provider = cfg["provider"] if "混合" in mode else "DeepSeek"
                futures.append(executor.submit(call_ai_api, f"市场情况：\n{stage2_text}", cfg["prompt"], target_provider, api_key_set, gemini_model))
             res = [f.result() for f in futures]
             st.session_state.analysis_results["risk_system"] = {"text": res[0], "provider": "DeepSeek"}
             st.session_state.analysis_results["risk_portfolio"] = {"text": res[1], "provider": "DeepSeek"}

    final_text = stage2_text + "\n" + res[0] + "\n" + res[1]
    with st.spinner("第四阶段：总经理正在决策..."):
        k = "general_manager"
        cfg = AGENTS_CONFIG[k]
        target_provider = cfg["provider"] if "混合" in mode else "DeepSeek"
        res = call_ai_api(f"所有报告：\n{final_text}", cfg["prompt"], target_provider, api_key_set, gemini_model)
        st.session_state.analysis_results[k] = {"text": res, "provider": target_provider}
    
    st.success("分析完成！")

def render_section(title, agent_keys, cols=1):
    st.subheader(title)
    columns = st.columns(cols)
    for i, key in enumerate(agent_keys):
        cfg = AGENTS_CONFIG[key]
        result_obj = st.session_state.analysis_results.get(key)
        
        content = result_obj["text"] if result_obj else "等待分析指令..."
        provider = result_obj["provider"] if result_obj else "Waiting"
        if provider == "Gemini": provider = gemini_model 
            
        border_color = "rgba(0,0,0,0.05)"
        if "risk" in key: border_color = "rgba(245, 158, 11, 0.4)"
        if "manager" in key: border_color = "rgba(139, 92, 246, 0.5)"
        if "general" in key: border_color = "#EF4444"

        with columns[i % cols]:
            st.markdown(f"""
            <div class="agent-card" style="border-color: {border_color};">
                <div class="card-header">
                    <div class="agent-info">
                        <img src="{cfg['avatar']}" class="avatar">
                        <div>
                            <div class="agent-name">{cfg['name']}</div>
                            <div class="agent-role">{cfg['role']}</div>
                        </div>
                    </div>
                    <span class="model-badge {provider.split('-')[0].lower()}">{provider}</span>
                </div>
                <div class="card-content">{content}</div>
            </div>
            """, unsafe_allow_html=True)

render_section("🔍 第一阶段：专业分析师", list(AGENTS_CONFIG.keys())[:5], cols=5)
render_section("🧠 第二阶段：策略整合", ["manager_fundamental", "manager_momentum"], cols=2)
render_section("🛡️ 第三阶段：风控评估", ["risk_system", "risk_portfolio"], cols=2)

gm_res = st.session_state.analysis_results.get("general_manager")
gm_text = gm_res["text"] if gm_res else "等待决策..."
gm_prov = gm_res["provider"] if gm_res else "Waiting"
st.markdown("---")
st.subheader("🏆 第四阶段：最终决议")
st.markdown(f"""
<div style="background: #FFFFFF; border: 2px solid #FF3B30; border-radius: 18px; padding: 30px; box-shadow: 0 10px 30px rgba(255, 59, 48, 0.1);">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; border-bottom:1px solid rgba(255, 59, 48, 0.1); padding-bottom:15px;">
        <div style="display:flex; align-items:center; gap:15px;">
            <img src="{AGENTS_CONFIG['general_manager']['avatar']}" style="width:60px; height:60px; border-radius:50%; border:2px solid #FF3B30;">
            <div>
                <span style="font-size:1.5em; font-weight:bold; color:#1D1D1F;">👑 投资决策总经理</span>
                <div style="color:#86868B; font-size:0.9em;">General Manager</div>
            </div>
        </div>
        <span style="background:#FFF1F2; color:#FF3B30; padding:4px 12px; border-radius:99px; font-size:0.8em; border:1px solid #FECACA;">{gm_prov}</span>
    </div>
    <div style="font-size:1.1em; line-height:1.8; color:#1D1D1F; white-space: pre-wrap;">{gm_text}</div>
</div>
""", unsafe_allow_html=True)