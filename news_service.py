"""
news_service.py — 新闻舆情服务
================================
为信号命中的股票搜索最近新闻，判断情绪。

数据源优先级:
  1. Tavily API（免费 1000次/月）
  2. 降级: 无新闻时返回 "neutral"

配置:
  环境变量 TAVILY_API_KEY=你的key
  获取: https://tavily.com/
"""

import os
import time
from typing import List, Dict

import requests


class NewsService:
    def __init__(self, news_max_age_days: int = 3):
        self.api_key = os.getenv("TAVILY_API_KEY", "").strip()
        self.news_max_age_days = news_max_age_days
        self._last_call = 0.0

    def _throttle(self):
        now = time.time()
        wait = 0.5 - (now - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def search_stock_news(self, symbol: str, name: str, max_results: int = 5) -> List[Dict]:
        if not self.api_key:
            return []
        self._throttle()
        payload = {
            "api_key": self.api_key,
            "query": f"{name} 股票 最新消息",
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
            "days": self.news_max_age_days,
        }
        try:
            resp = requests.post("https://api.tavily.com/search", json=payload, timeout=12)
            resp.raise_for_status()
            data = resp.json() or {}
            items = data.get("results", []) or []
            normalized = []
            for it in items:
                normalized.append({
                    "title": it.get("title", ""),
                    "content": it.get("content", ""),
                    "url": it.get("url", ""),
                    "published_date": it.get("published_date") or it.get("published_time") or "",
                })
            return normalized
        except Exception:
            return []

    def analyze_sentiment(self, news_list: List[Dict], stock_name: str) -> Dict:
        if not news_list:
            return {
                "sentiment": "neutral",
                "score": 0,
                "summary": "无近期新闻",
                "news_count": 0,
                "headlines": [],
            }

        positive_keywords = [
            "利好", "增长", "突破", "创新高", "政策支持", "业绩超预期",
            "回购", "增持", "分红", "扩产", "订单", "中标",
        ]
        negative_keywords = [
            "利空", "下跌", "暴雷", "违规", "处罚", "减持", "业绩下滑",
            "亏损", "退市", "诉讼", "调查", "暂停",
        ]

        text = " ".join((it.get("title", "") + " " + it.get("content", "")) for it in news_list)
        pos_hits = [kw for kw in positive_keywords if kw in text]
        neg_hits = [kw for kw in negative_keywords if kw in text]
        pos_count = len(pos_hits)
        neg_count = len(neg_hits)

        if neg_count > pos_count:
            sentiment = "negative"
            score = -20
        elif pos_count > neg_count:
            sentiment = "positive"
            score = 10
        else:
            sentiment = "neutral"
            score = 0

        pos_part = "/".join(pos_hits[:2]) if pos_hits else "无"
        neg_part = "/".join(neg_hits[:2]) if neg_hits else "无"
        summary = f"近3天{pos_count}条利好({pos_part})、{neg_count}条利空({neg_part})"
        headlines = [it.get("title", "") for it in news_list[:3] if it.get("title")]
        return {
            "sentiment": sentiment,
            "score": score,
            "summary": summary,
            "news_count": len(news_list),
            "headlines": headlines,
        }

    def enrich_signals(self, signals: List[Dict]) -> List[Dict]:
        enriched = []
        for sig in signals or []:
            symbol = sig.get("symbol", "")
            name = sig.get("name", symbol)
            news = self.search_stock_news(symbol, name)
            senti = self.analyze_sentiment(news, name)
            score_adj = int(senti.get("score", 0) or 0)
            base_score = float(sig.get("score", 0) or 0)
            sig["score"] = int(max(0, min(100, round(base_score + score_adj))))
            sig["news"] = {
                "sentiment": senti.get("sentiment", "neutral"),
                "score_adj": score_adj,
                "summary": senti.get("summary", "无近期新闻"),
                "headlines": senti.get("headlines", []),
            }
            enriched.append(sig)
        return enriched
