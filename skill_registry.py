"""
skill_registry.py — 各 skill 模块的 ToolCenter 注册
=====================================================

接入方式有两种，选你喜欢的：
  方式 A: 在每个 skill 模块内部自行注册（推荐，注册逻辑跟着 skill 走）
  方式 B: 集中在一个文件里注册（本文件，适合过渡期）

过渡期建议用方式 B，稳定后逐步迁移到方式 A。

使用：
    # 在系统启动时（如 main.py / daily_agent.py 入口处）
    from skill_registry import register_all_skills
    register_all_skills()
"""

from tool_center import ToolCenter


def register_all_skills():
    """
    集中注册所有 skill（方式 B — 过渡期使用）

    后续迁移时，把每个 register 调用移到对应的 skill 模块内部即可。
    """

    # ─────────────────────────────────────────
    # 分析类 (analysis)
    # ─────────────────────────────────────────

    ToolCenter.register(
        name="sentiment_analysis",
        func=_get_skill_func("sentiment_analysis"),
        description="分析A股市场情绪：新闻舆情、资金流向、市场恐慌/贪婪指数综合评估",
        category="analysis",
        keywords=["情绪", "舆情", "恐慌", "贪婪", "市场情绪", "sentiment",
                  "新闻", "消息面", "利好", "利空", "北向", "资金流"],
        examples=["当前市场情绪如何", "今天有什么重大消息", "北向资金怎么样"],
        priority=5,
    )

    ToolCenter.register(
        name="sector_rotation",
        func=_get_skill_func("sector_rotation"),
        description="分析A股板块轮动：板块强弱排名、轮动方向预判、热点切换信号识别",
        category="analysis",
        keywords=["板块", "轮动", "行业", "热点", "切换", "sector", "主线",
                  "龙头", "题材", "概念", "赛道"],
        examples=["当前哪个板块最强", "板块轮动到什么阶段了", "主线是什么"],
        priority=5,
    )

    ToolCenter.register(
        name="canslim_screening",
        func=_get_skill_func("canslim_screening"),
        description="CANSLIM选股法：按C(当季EPS)/A(年度EPS)/N(新产品)/S(供需)/L(领涨)/I(机构)/M(大盘)七因子筛选A股标的",
        category="analysis",
        keywords=["CANSLIM", "选股", "筛选", "EPS", "基本面", "成长股",
                  "CAN SLIM", "欧奈尔", "screening"],
        examples=["用CANSLIM帮我选几只股票", "哪些股票基本面好", "成长股推荐"],
        priority=4,
    )

    ToolCenter.register(
        name="technical_analysis",
        func=_get_skill_func("technical_analysis"),
        description="技术面分析：K线形态、均线系统、MACD/KDJ/RSI等指标、量价关系、支撑阻力位",
        category="analysis",
        keywords=["技术", "K线", "均线", "MACD", "KDJ", "RSI", "量价",
                  "支撑", "阻力", "突破", "形态", "缩量", "放量", "金叉", "死叉",
                  "涨停", "跌停", "连板", "技术面"],
        examples=["帮我看看000001的技术面", "这只股票的支撑位在哪", "MACD什么信号"],
        priority=4,
    )

    ToolCenter.register(
        name="backtest",
        func=_get_skill_func("backtest"),
        description="策略回测：对指定策略和标的池进行历史数据回测，输出收益、回撤、胜率等指标",
        category="analysis",
        keywords=["回测", "backtest", "历史", "胜率", "收益率", "回撤",
                  "夏普", "策略验证"],
        examples=["帮我回测一下板块轮动策略", "这个策略历史表现怎么样"],
        priority=3,
    )

    # ─────────────────────────────────────────
    # 交易类 (trading)
    # ─────────────────────────────────────────

    ToolCenter.register(
        name="trade_signals",
        func=_get_skill_func("trade_signals"),
        description="生成交易信号：综合分析后给出买入/卖出/观望信号及具体操作建议",
        category="trading",
        keywords=["交易", "买入", "卖出", "信号", "操作", "建仓", "加仓",
                  "减仓", "清仓", "止损", "止盈", "买什么", "卖什么"],
        examples=["今天有什么交易信号", "该买入还是观望", "给我操作建议"],
        priority=5,
    )

    ToolCenter.register(
        name="position_manager",
        func=_get_skill_func("position_manager"),
        description="持仓管理：查看当前持仓、仓位分布、盈亏情况、持仓建议",
        category="trading",
        keywords=["持仓", "仓位", "盈亏", "浮盈", "浮亏", "持股",
                  "portfolio", "仓位管理"],
        examples=["看看我的持仓", "当前仓位分布", "哪只股票亏了"],
        priority=4,
    )

    # ─────────────────────────────────────────
    # 风控类 (risk)
    # ─────────────────────────────────────────

    ToolCenter.register(
        name="risk_check",
        func=_get_skill_func("risk_check"),
        description="风控检查：止损线监控、持仓集中度、最大回撤、涨跌停风险、仓位上限检查",
        category="risk",
        keywords=["风控", "风险", "止损", "回撤", "集中度", "仓位上限",
                  "risk", "风险管理"],
        examples=["帮我做风控检查", "有没有需要止损的", "当前风险怎么样"],
        priority=6,  # 风控优先级最高
    )

    # ─────────────────────────────────────────
    # 综合/workflow 类
    # ─────────────────────────────────────────

    ToolCenter.register(
        name="daily_workflow",
        func=_get_skill_func("daily_workflow"),
        description="每日工作流：完整的盘前扫描→盘中监控→盘后复盘流程",
        category="system",
        keywords=["复盘", "总结", "工作流", "每日", "日报", "workflow",
                  "盘前", "盘后", "早盘", "尾盘"],
        examples=["帮我做盘后复盘", "今天总结一下", "盘前准备"],
        priority=3,
    )

    ToolCenter.register(
        name="market_overview",
        func=_get_skill_func("market_overview"),
        description="大盘概览：指数走势、涨跌家数、成交额、北向资金、市场温度计",
        category="analysis",
        keywords=["大盘", "指数", "上证", "深证", "创业板", "涨跌",
                  "成交额", "市场", "overview", "行情"],
        examples=["大盘怎么样", "今天行情如何", "上证多少点了"],
        priority=5,
    )


def _get_skill_func(skill_name: str) -> callable:
    """
    延迟获取 skill 函数引用

    避免注册时就 import 所有 skill 模块（有些可能还没加载）。
    实际执行时才通过 Router 调用，所以这里返回一个 placeholder。

    当你逐步迁移到方式 A（每个 skill 自行注册）时，这个函数就不需要了。
    """
    def placeholder(**kwargs):
        raise NotImplementedError(
            f"Skill '{skill_name}' 的函数引用尚未绑定。\n"
            f"请在对应的 skill 模块中使用 ToolCenter.register() 传入实际函数，\n"
            f"或在此文件中替换 _get_skill_func('{skill_name}') 为实际函数引用。"
        )
    placeholder.__name__ = skill_name
    return placeholder


# ═══════════════════════════════════════════
# 方式 A 示例：在 skill 模块内部自行注册
# ═══════════════════════════════════════════

"""
# ── 文件: skills/sector_rotation.py ──

from tool_center import tool

@tool(
    name="sector_rotation",
    description="分析A股板块轮动：板块强弱排名、轮动方向预判、热点切换信号识别",
    category="analysis",
    keywords=["板块", "轮动", "行业", "热点", "切换", "主线", "龙头"],
    examples=["当前哪个板块最强", "板块轮动到什么阶段了"],
    priority=5,
)
def analyze_sector_rotation(query: str = "", **kwargs) -> str:
    # ... 你现有的板块轮动分析代码 ...
    pass


# ── 文件: skills/sentiment_analysis.py ──

from tool_center import tool

@tool(
    name="sentiment_analysis",
    description="分析A股市场情绪：舆情、资金流向、恐慌贪婪指数",
    category="analysis",
    keywords=["情绪", "舆情", "恐慌", "贪婪", "消息面", "北向"],
    examples=["当前市场情绪如何", "今天有什么重大消息"],
    priority=5,
)
def analyze_sentiment(query: str = "", **kwargs) -> str:
    # ... 你现有的情绪分析代码 ...
    pass
"""
