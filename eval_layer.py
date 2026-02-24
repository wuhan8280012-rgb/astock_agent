"""
Eval层: 自动验证Skill准确性

核心思路 (参考 Dexter):
  不是LLM-as-judge打分，而是用真实市场数据验证:
  - 情绪Skill说"恐慌" → 后续5日市场真的跌了吗？
  - 板块Skill说"电子强势" → 电子板块后续真的跑赢大盘了吗？
  - 宏观Skill说"流动性紧张" → SHIBOR真的上行了吗？

自动化流程:
  1. 从 scratchpad 读取历史预测
  2. 拉取预测之后的实际行情
  3. 计算命中率
  4. 输出每个Skill的"可信度"评分

用法:
  python eval_layer.py                    # 评估所有Skill
  python eval_layer.py --skill sentiment  # 只评估情绪Skill
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import get_fetcher
from scratchpad import Scratchpad


class EvalLayer:
    """Skill准确性评估"""

    def __init__(self):
        self.fetcher = get_fetcher()
        self.sp = Scratchpad()
        self.eval_path = Path("./knowledge/eval_results.json")
        self.eval_path.parent.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> Dict:
        """评估所有Skill"""
        results = {}

        print("[Eval] 评估情绪Skill...")
        results["sentiment"] = self.eval_sentiment()

        print("[Eval] 评估板块轮动Skill...")
        results["sector"] = self.eval_sector()

        print("[Eval] 评估宏观Skill...")
        results["macro"] = self.eval_macro()

        # 保存结果
        self._save_results(results)
        return results

    # ========================================================
    # 情绪Skill评估
    # ========================================================
    def eval_sentiment(self) -> Dict:
        """
        情绪Skill说"恐慌/贪婪"时，后续市场走势是否符合？

        规则:
          恐慌/极度恐慌 → 预期后续5日下跌 → 实际下跌=命中
          贪婪/极度贪婪 → 预期后续5日上涨 → 实际上涨=命中
          中性 → 不计入统计
        """
        entries = self.sp.query(skill="sentiment")
        if not entries:
            return {"samples": 0, "message": "无历史数据"}

        hits = 0
        total = 0
        details = []

        for entry in entries:
            output = entry.get("output", {})
            level = output.get("overall_level", "")
            date = output.get("date", "")

            if not date or level in ["中性", "数据不足", ""]:
                continue

            # 获取后续5日涨跌
            ret_5d = self._get_index_return(date, 5)
            if ret_5d is None:
                continue

            # 判断命中
            if level in ["恐慌", "极度恐慌"]:
                expected = "下跌"
                hit = ret_5d < 0
            elif level in ["贪婪", "极度贪婪"]:
                expected = "上涨"
                hit = ret_5d > 0
            else:
                continue

            total += 1
            if hit:
                hits += 1

            details.append({
                "date": date,
                "prediction": f"{level}→预期{expected}",
                "actual_5d_ret": round(ret_5d, 2),
                "hit": hit,
            })

        accuracy = round(hits / total * 100, 1) if total > 0 else 0

        return {
            "skill": "sentiment",
            "samples": total,
            "hits": hits,
            "accuracy": accuracy,
            "grade": self._grade(accuracy, total),
            "details": details[-10:],  # 只保留最近10条
        }

    # ========================================================
    # 板块轮动评估
    # ========================================================
    def eval_sector(self) -> Dict:
        """
        板块Skill推荐的Top5板块，后续10日是否跑赢中证500？

        规则:
          Top5板块的平均10日涨幅 > 中证500的10日涨幅 = 命中
        """
        entries = self.sp.query(skill="sector_rotation")
        if not entries:
            return {"samples": 0, "message": "无历史数据"}

        hits = 0
        total = 0
        details = []

        for entry in entries:
            output = entry.get("output", {})
            top_sectors = output.get("top_sectors", [])
            date = output.get("date", "")

            if not date or not top_sectors:
                continue

            # 中证500作为基准
            benchmark_ret = self._get_index_return(date, 10, index_code="000905.SH")
            if benchmark_ret is None:
                continue

            # Top板块的平均涨幅（简化：用板块名匹配，从sector数据获取）
            sector_rets = []
            for s in top_sectors[:5]:
                name = s.get("name", "")
                ret = self._get_sector_return(name, date, 10)
                if ret is not None:
                    sector_rets.append(ret)

            if not sector_rets:
                continue

            avg_sector_ret = np.mean(sector_rets)
            hit = avg_sector_ret > benchmark_ret

            total += 1
            if hit:
                hits += 1

            details.append({
                "date": date,
                "top_sectors": [s.get("name", "") for s in top_sectors[:5]],
                "avg_sector_10d": round(avg_sector_ret, 2),
                "benchmark_10d": round(benchmark_ret, 2),
                "excess": round(avg_sector_ret - benchmark_ret, 2),
                "hit": hit,
            })

        accuracy = round(hits / total * 100, 1) if total > 0 else 0

        return {
            "skill": "sector_rotation",
            "samples": total,
            "hits": hits,
            "accuracy": accuracy,
            "grade": self._grade(accuracy, total),
            "details": details[-10:],
        }

    # ========================================================
    # 宏观Skill评估
    # ========================================================
    def eval_macro(self) -> Dict:
        """
        宏观Skill的流动性判断是否与后续市场走势一致？

        规则:
          宽松 → 预期后续10日上涨 → 实际上涨=命中
          紧张 → 预期后续10日下跌 → 实际下跌=命中
        """
        entries = self.sp.query(skill="macro")
        if not entries:
            return {"samples": 0, "message": "无历史数据"}

        hits = 0
        total = 0
        details = []

        for entry in entries:
            output = entry.get("output", {})
            level = output.get("liquidity_level", "")
            date = output.get("date", "")

            if not date or "中性" in level or not level:
                continue

            ret_10d = self._get_index_return(date, 10)
            if ret_10d is None:
                continue

            if level in ["宽松", "中性偏松"]:
                expected = "上涨"
                hit = ret_10d > 0
            elif level in ["紧张", "中性偏紧"]:
                expected = "下跌"
                hit = ret_10d < 0
            else:
                continue

            total += 1
            if hit:
                hits += 1

            details.append({
                "date": date,
                "prediction": f"{level}→预期{expected}",
                "actual_10d_ret": round(ret_10d, 2),
                "hit": hit,
            })

        accuracy = round(hits / total * 100, 1) if total > 0 else 0

        return {
            "skill": "macro",
            "samples": total,
            "hits": hits,
            "accuracy": accuracy,
            "grade": self._grade(accuracy, total),
            "details": details[-10:],
        }

    # ========================================================
    # 工具方法
    # ========================================================
    def _get_index_return(self, start_date: str, days: int,
                          index_code: str = "000300.SH") -> Optional[float]:
        """获取指数从start_date起的N日收益率"""
        try:
            df = self.fetcher.get_index_daily(index_code, days=days + 30)
            if df.empty:
                return None
            df = df.sort_values("trade_date").reset_index(drop=True)
            # 找到start_date之后的数据
            after = df[df["trade_date"] > start_date]
            if len(after) < days:
                return None
            start_close = df[df["trade_date"] <= start_date].iloc[-1]["close"]
            end_close = after.iloc[days - 1]["close"]
            return (end_close / start_close - 1) * 100
        except Exception:
            return None

    def _get_sector_return(self, sector_name: str, start_date: str,
                           days: int) -> Optional[float]:
        """获取板块从start_date起的N日收益率"""
        try:
            from config import SW_SECTORS
            # 找到板块代码
            code = None
            for c, n in SW_SECTORS.items():
                if n == sector_name:
                    code = c
                    break
            if not code:
                return None

            df = self.fetcher.get_sector_daily(code, days=days + 30)
            if df.empty:
                return None
            df = df.sort_values("trade_date").reset_index(drop=True)
            after = df[df["trade_date"] > start_date]
            if len(after) < days:
                return None
            start_close = df[df["trade_date"] <= start_date].iloc[-1]["close"]
            end_close = after.iloc[days - 1]["close"]
            return (end_close / start_close - 1) * 100
        except Exception:
            return None

    def _grade(self, accuracy: float, samples: int) -> str:
        """给Skill打等级"""
        if samples < 5:
            return "样本不足"
        if accuracy >= 70:
            return "A (优秀)"
        elif accuracy >= 55:
            return "B (良好)"
        elif accuracy >= 40:
            return "C (一般)"
        else:
            return "D (需优化)"

    def _save_results(self, results: Dict):
        data = {
            "eval_time": datetime.now().isoformat(),
            "results": results,
        }
        with open(self.eval_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ========================================================
    # 报告输出
    # ========================================================
    def print_report(self, results: Dict):
        print(f"\n{'='*60}")
        print(f"  📊 Skill 准确性评估报告")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")

        for skill_name, r in results.items():
            samples = r.get("samples", 0)
            if samples == 0:
                print(f"\n  [{skill_name}] {r.get('message', '无数据')}")
                continue

            acc = r.get("accuracy", 0)
            grade = r.get("grade", "?")
            print(f"\n  [{skill_name}] 样本{samples} | 命中{r.get('hits',0)} | "
                  f"准确率{acc}% | 等级: {grade}")

            # 最近几条详情
            for d in r.get("details", [])[-5:]:
                hit_icon = "✅" if d.get("hit") else "❌"
                pred = d.get("prediction", "")
                actual = d.get("actual_5d_ret") or d.get("actual_10d_ret") or 0
                print(f"    {hit_icon} {d.get('date','')} {pred} → 实际{actual:+.2f}%")

        print(f"\n{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", type=str, default=None)
    args = parser.parse_args()

    evaluator = EvalLayer()

    if args.skill:
        method = getattr(evaluator, f"eval_{args.skill}", None)
        if method:
            result = method()
            evaluator.print_report({args.skill: result})
        else:
            print(f"未知Skill: {args.skill}")
    else:
        results = evaluator.run_all()
        evaluator.print_report(results)
