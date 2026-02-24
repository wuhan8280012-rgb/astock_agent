"""
config_deepseek.py — DeepSeek API 配置补丁
===========================================
把以下内容加到你的 config.py 中即可。
"""

# ═══════════════════════════════════════════
# DeepSeek API（加到你的 config.py）
# ═══════════════════════════════════════════

# API Key（建议改用环境变量，见下方说明）
DEEPSEEK_API_KEY = "sk-xxx"  # ← 替换为你的真实 key

# 更安全的做法：用环境变量，不把 key 写进代码
# 在终端执行：export DEEPSEEK_API_KEY=sk-xxx
# 或在 .env 文件中写入（记得 .gitignore 加上 .env）

# ═══════════════════════════════════════════
# 推荐的安全做法
# ═══════════════════════════════════════════

SETUP_GUIDE = """
方式 1（推荐）: 环境变量
────────────────────────
# Linux/Mac: 加到 ~/.bashrc 或 ~/.zshrc
export DEEPSEEK_API_KEY=sk-xxx

# 或创建 .env 文件（项目根目录）
echo 'DEEPSEEK_API_KEY=sk-xxx' > .env

# 然后在 config.py 中：
import os
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


方式 2: 直接写在 config.py（简单但不够安全）
────────────────────────────────────────────
# config.py 中直接写：
DEEPSEEK_API_KEY = "sk-xxx"
# ⚠️ 确保 config.py 在 .gitignore 中，不要提交到 git


方式 3: 命令行传参
────────────────────────
python3 router.py "问题" --api-key sk-xxx
"""


# ═══════════════════════════════════════════
# Router 对接改动（你的 router.py）
# ═══════════════════════════════════════════

ROUTER_PATCH = """
# ---- router.py 改动 ----

# 1. 顶部加一行
from llm_client import get_llm
llm = get_llm()

# 2. 在 _route_with_llm() 中，把原来调用 Anthropic API 的代码：
#    response = anthropic_client.messages.create(...)
#    改为：
response = llm.call_for_router(messages)
#    messages 格式不变（都是 [{"role": "...", "content": "..."}]）

# 3. 在需要深度分析的地方（如盘后复盘），用 reason()：
result = llm.reason("请深度分析当前板块轮动...")

# 4.（可选）在 Skill 执行中需要 LLM 时：
result = llm.chat(prompt, system="你是A股量化分析师...")
"""
