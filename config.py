# ==========================================
# 配置文件 (config.py)
# ==========================================

# 1. Google Gemini 配置
# 申请地址: https://aistudio.google.com/app/apikey
GEMINI_API_KEY = "AIzaSyACvbIu1kHdWpQ6qSMTvSplXoPTjQv_PiI"

# 模型版本 (根据您提供的截图更新)
# 选项 A: "gemini-2.5-flash" (推荐：速度极快，最新稳定版)
# 选项 B: "gemini-2.5-pro" (推理更强，但可能消耗更多额度)
GEMINI_MODEL_NAME = "gemini-2.5-flash" 

# 2. DeepSeek 配置
# 申请地址: https://platform.deepseek.com/
DEEPSEEK_API_KEY = "sk-3c459e9dbc2945c29ce51f990e69a997"
DEEPSEEK_MODEL_NAME = "deepseek-chat"

# 3. 阿里云通义千问 配置
# 申请地址: https://dashscope.console.aliyun.com/
QWEN_API_KEY = "sk-be77fc253e604ea1bfd0a926859f4fd3"
QWEN_MODEL_NAME = "qwen-plus"