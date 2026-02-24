# Lessons Learned
- Tushare北向资金字段名不统一：优先用 north_money，fallback 到 hgt+sgt
- API频率：批量调用间加 time.sleep(0.1)
- DataFrame 计算前必须检查 NaN
- 所有日期计算用交易日历，不用自然日
