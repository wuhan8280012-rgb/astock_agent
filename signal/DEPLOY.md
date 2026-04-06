# 动量轮动策略 · 半自动实盘部署指南

## 一、前置条件

### 1. Python 环境
```bash
# Python 3.10+
python --version

# 安装依赖
pip install pandas numpy tushare
```

### 2. Tushare API
- 注册 https://tushare.pro 并获取 token
- 建议积分 >= 2000 (确保有中证1000成分股权限)

### 3. 广发证券账户
- 已开通广发易淘金APP
- 后续升级可申请 miniQMT 量化交易接口

---

## 二、安装配置

### 1. 克隆项目
```bash
cd ~/your-workspace
# 将 stock_agent/new/ 目录复制到你的机器上
```

### 2. 配置 Tushare Token
编辑 `config/.env`:
```
TUSHARE_TOKEN=你的tushare_token
```

### 3. 配置推送渠道
编辑 `config/notify_config.json`，至少配置一个渠道：

#### QQ邮箱推送 (推荐)
```json
{
  "email_enabled": true,
  "smtp_host": "smtp.qq.com",
  "smtp_port": 465,
  "smtp_user": "你的QQ号@qq.com",
  "smtp_password": "QQ邮箱授权码",
  "email_to": ["你的接收邮箱@xxx.com"]
}
```
获取QQ邮箱授权码: QQ邮箱设置 → 账户 → POP3/SMTP服务 → 开启 → 生成授权码

#### 钉钉机器人
```json
{
  "dingtalk_enabled": true,
  "dingtalk_webhook": "https://oapi.dingtalk.com/robot/send?access_token=xxx"
}
```
创建方式: 钉钉群 → 设置 → 智能群助手 → 添加自定义机器人

#### Server酱 (微信推送)
```json
{
  "serverchan_enabled": true,
  "serverchan_key": "你的key"
}
```
注册: https://sct.ftqq.com

### 4. 初始化持仓
```bash
python signal/run_daily_signal.py --init
# 输入你的初始资金金额
```

### 5. 测试推送
```bash
python signal/run_daily_signal.py --test-notify
```

---

## 三、每日使用流程

### 收盘后 (15:00-15:30)

```bash
# 1. 生成今日信号
python signal/run_daily_signal.py
```

系统将自动：
- 拉取中证1000最新行情
- 计算动量排名
- 检查止损
- 判断是否调仓日
- 生成买卖信号
- 推送到你配置的渠道

### 收到信号后

1. 打开广发易淘金APP
2. 按信号执行买卖 (优先执行止损信号!)
3. 记录实际成交价

### 执行完毕后

```bash
# 2. 确认交易执行, 更新持仓
python signal/run_daily_signal.py --confirm 20260320
# 按提示输入 y (全部确认) 或 c (逐条确认, 可输入实际成交价)
```

### 查看状态

```bash
python signal/run_daily_signal.py --status
```

---

## 四、定时自动运行 (推荐)

### macOS / Linux
```bash
# 编辑 crontab
crontab -e

# 每个交易日 15:30 自动运行
30 15 * * 1-5 cd /path/to/stock_agent/new && /usr/bin/python3 signal/run_daily_signal.py >> logs/cron.log 2>&1
```

### Windows
```batch
:: 创建 run_signal.bat
@echo off
cd /d C:\path\to\stock_agent\new
python signal\run_daily_signal.py >> logs\cron.log 2>&1
```
任务计划程序 → 创建基本任务 → 每天15:30运行 → 选择上述bat文件

---

## 五、使用离线数据 (无tushare积分时)

如果 tushare 积分不够无法自动拉取数据, 可手动准备CSV:

```bash
# 使用CSV文件
python signal/run_daily_signal.py --csv data/csi1000_market_bundle.csv --date 20260320
```

CSV格式要求: 参见 `data/csi1000_market_bundle.csv` 的结构

---

## 六、文件结构说明

```
signal/
├── signal_generator.py   # 信号生成核心引擎
├── notifier.py           # 多渠道消息推送
├── run_daily_signal.py   # 每日一键运行入口
└── DEPLOY.md             # 本文档

config/
├── .env                  # API密钥配置
└── notify_config.json    # 推送渠道配置

data/
├── portfolio_state.json  # 持仓状态 (自动维护)
└── signals/              # 历史信号存档
    └── signal_YYYYMMDD.json

logs/
└── signal_YYYYMMDD.txt   # 信号报告日志
```

---

## 七、常见问题

**Q: tushare 返回空数据怎么办?**
A: 检查积分是否 >= 2000, 或改用离线CSV模式。

**Q: 广发有自动下单接口吗?**
A: 联系客户经理申请 miniQMT, 开通后可升级到全自动模式。

**Q: 止损信号一定要执行吗?**
A: 强烈建议执行。回测数据显示 -8% 止损是该策略的核心优势之一。

**Q: 非调仓日有信号怎么办?**
A: 只有止损信号需要立即执行, 其他的可以等到调仓日。

**Q: 如何从现有持仓开始?**
A: 编辑 `data/portfolio_state.json`, 在 positions 中添加现有持仓:
```json
{
  "positions": {
    "300672.SZ": {
      "shares": 700,
      "cost_price": 185.50,
      "entry_date": "20260315",
      "peak_price": 195.10,
      "current_price": 195.10,
      "name": "国科微"
    }
  }
}
```
