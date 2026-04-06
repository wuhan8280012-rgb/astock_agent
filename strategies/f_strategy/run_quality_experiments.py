#!/usr/bin/env python3
"""
Quality-filter experiments for F strategy on CSI 1000 5y.
"""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / 'data_exports' / 'tushare_20210329_20260327_csi1000_5y' / 'csi1000_market_bundle_5y.csv'
BT_SCRIPT = PROJECT_ROOT / 'scripts' / 'backtest_strategies.py'
OUT_PATH = PROJECT_ROOT / 'backtest' / 'strategy_f_quality_experiments_csi1000_5y.json'


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dataset(data_path: Path):
    raw = pd.read_csv(data_path, low_memory=False)
    for col in ['trade_date', 'list_date', 'cal_date']:
        if col in raw.columns:
            raw[col] = raw[col].astype(str).str.replace(r'\\.0$', '', regex=True).str.zfill(8)

    daily_cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg']
    if 'circ_mv' in raw.columns:
        daily_cols.append('circ_mv')
    daily = raw[raw['data_type'] == 'daily'][daily_cols].copy()
    for col in [c for c in daily.columns if c not in {'ts_code', 'trade_date'}]:
        daily[col] = pd.to_numeric(daily[col], errors='coerce')
    daily = daily.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    adj = raw[raw['data_type'] == 'adj_factor'][['ts_code', 'trade_date', 'adj_factor']].copy()
    adj['adj_factor'] = pd.to_numeric(adj['adj_factor'], errors='coerce')
    daily = daily.merge(adj, on=['ts_code', 'trade_date'], how='left')
    daily['adj_close'] = daily['close'] * daily['adj_factor']

    db = raw[raw['data_type'] == 'daily_basic'][['ts_code', 'trade_date', 'total_mv', 'circ_mv']].copy()
    for col in ['total_mv', 'circ_mv']:
        db[col] = pd.to_numeric(db[col], errors='coerce')
    daily = daily.merge(db, on=['ts_code', 'trade_date'], how='left', suffixes=('', '_db'))
    if 'circ_mv_db' in daily.columns:
        daily['circ_mv'] = daily['circ_mv_db'].combine_first(daily.get('circ_mv'))
        daily = daily.drop(columns=['circ_mv_db'])
    if 'total_mv_db' in daily.columns:
        daily['total_mv'] = daily['total_mv_db']
        daily = daily.drop(columns=['total_mv_db'])

    idx = raw[raw['data_type'] == 'index_daily'][['trade_date', 'close', 'pct_chg']].copy()
    idx.columns = ['trade_date', 'idx_close', 'idx_pct_chg']
    for col in ['idx_close', 'idx_pct_chg']:
        idx[col] = pd.to_numeric(idx[col], errors='coerce')
    idx = idx.sort_values('trade_date').reset_index(drop=True)

    basic = raw[raw['data_type'] == 'stock_basic'][['ts_code', 'name', 'industry', 'list_date']].copy()
    trade_dates = sorted(daily['trade_date'].unique())
    return daily, idx, basic, trade_dates


def calc_max_drawdown(closes: np.ndarray) -> float:
    peaks = np.maximum.accumulate(closes)
    drawdowns = closes / peaks - 1
    return float(drawdowns.min())


def make_quality_backtest(module, filter_name: str):
    class QualityBacktest(module.Backtest):
        def _score_universe(self, date: str):
            cfg = self.cfg
            if date not in self.trade_dates:
                return []

            scores = []
            for code, data in self._stock_data.items():
                if date not in data.index:
                    continue
                row = data.loc[date]

                close = row['close']
                if pd.isna(close) or close < cfg.min_price:
                    continue

                info = self._basic_map.get(code)
                if info is not None:
                    name = str(info.get('name', ''))
                    if 'ST' in name.upper():
                        continue
                    list_date = str(info.get('list_date', ''))
                    if list_date and len(list_date) >= 8:
                        try:
                            from datetime import datetime
                            ld = datetime.strptime(list_date[:8], '%Y%m%d')
                            cd = datetime.strptime(date, '%Y%m%d')
                            if (cd - ld).days < cfg.min_list_days:
                                continue
                        except Exception:
                            pass

                hist = data[data.index <= date]
                if len(hist) < 65:
                    continue

                recent_20 = hist.tail(20)
                avg_amount_20 = recent_20['amount'].mean() * 1000
                if avg_amount_20 < cfg.min_amount_20d:
                    continue

                if row['pct_chg'] >= 9.5:
                    continue

                closes = hist['close'].values.astype(float)
                amounts = hist['amount'].values.astype(float)

                mom_score = closes[-1] / closes[-61] - 1
                if np.isnan(mom_score):
                    continue

                vol_component = 0.0
                if cfg.use_volatility_factor and len(closes) >= cfg.volatility_days + 1:
                    rets = np.diff(closes[-cfg.volatility_days - 1:]) / closes[-cfg.volatility_days - 1:-1]
                    vol = np.std(rets)
                    if vol > 0:
                        vol_component = -vol

                size_component = 0.0
                if cfg.use_size_factor:
                    circ_mv = row.get('circ_mv', None)
                    if circ_mv and not pd.isna(circ_mv) and circ_mv > 0:
                        size_component = -np.log(circ_mv)

                if filter_name in {'price_structure', 'combined_light'}:
                    ma20 = np.mean(closes[-20:])
                    ma60 = np.mean(closes[-60:])
                    if not (close > ma20 and close > ma60):
                        continue

                if filter_name in {'drawdown_quality', 'combined_light'}:
                    max_dd_60 = calc_max_drawdown(closes[-60:])
                    if max_dd_60 < -0.18:
                        continue

                if filter_name in {'vol_compression', 'combined_light'}:
                    ret10 = np.diff(closes[-11:]) / closes[-11:-1]
                    ret20 = np.diff(closes[-21:]) / closes[-21:-1]
                    vol10 = np.std(ret10)
                    vol20 = np.std(ret20)
                    if not (vol10 < vol20 * 0.85):
                        continue

                if filter_name in {'volume_quality', 'combined_light'}:
                    avg_amount_60 = amounts[-60:].mean() * 1000
                    last_amount = amounts[-1] * 1000
                    tail20 = hist.tail(20)
                    up_amount = tail20.loc[tail20['pct_chg'] > 0, 'amount'].mean() * 1000
                    down_amount = tail20.loc[tail20['pct_chg'] <= 0, 'amount'].mean() * 1000
                    if np.isnan(up_amount):
                        up_amount = 0.0
                    if np.isnan(down_amount):
                        down_amount = 0.0
                    if avg_amount_20 < avg_amount_60 * 0.8:
                        continue
                    if last_amount > avg_amount_20 * 3.0:
                        continue
                    if up_amount < down_amount:
                        continue

                scores.append((code, mom_score, vol_component, size_component))

            if not scores:
                return []

            df = pd.DataFrame(scores, columns=['code', 'mom', 'vol', 'size'])
            df['mom_rank'] = df['mom'].rank(ascending=False)
            df['vol_rank'] = df['vol'].rank(ascending=False)
            df['size_rank'] = df['size'].rank(ascending=False)

            mom_weight = 1.0
            vol_w = cfg.volatility_weight if cfg.use_volatility_factor else 0
            size_w = cfg.size_weight if cfg.use_size_factor else 0
            total_w = mom_weight + vol_w + size_w

            df['composite_rank'] = (
                (mom_weight / total_w) * df['mom_rank'] +
                (vol_w / total_w) * df['vol_rank'] +
                (size_w / total_w) * df['size_rank']
            )
            df = df.sort_values('composite_rank').reset_index(drop=True)
            return list(zip(df['code'], df['composite_rank']))

    return QualityBacktest


def main():
    module = load_module('backtest_strategies', BT_SCRIPT)
    daily, idx, basic, trade_dates = load_dataset(DATA_PATH)
    cfg = [s for s in module.get_strategies() if s.name == 'F_三因子+趋势过滤'][0]

    experiments = [
        ('baseline', '原始 F', module.Backtest),
        ('price_structure', '价格结构: close > MA20 且 > MA60', make_quality_backtest(module, 'price_structure')),
        ('drawdown_quality', '回撤质量: 60日最大回撤 >= -18%', make_quality_backtest(module, 'drawdown_quality')),
        ('vol_compression', '波动压缩: 10日波动率 < 20日波动率 * 0.85', make_quality_backtest(module, 'vol_compression')),
        ('volume_quality', '放量质量: 避免极端爆量, 上涨量能优于下跌', make_quality_backtest(module, 'volume_quality')),
        ('combined_light', '组合轻过滤', make_quality_backtest(module, 'combined_light')),
    ]

    results = []
    for key, desc, cls in experiments:
        print(f'RUN {key}', flush=True)
        t0 = time.time()
        bt = cls(cfg, daily, idx, basic, trade_dates)
        result = bt.run(start_offset=250)
        result['elapsed_sec'] = round(time.time() - t0, 1)
        results.append({'experiment': key, 'rule': desc, **result})
        print(f"DONE {key} {result['total_return_pct']} {result['annual_return_pct']} {result['sharpe']} {result['max_drawdown_pct']}", flush=True)

    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'WROTE {OUT_PATH}')


if __name__ == '__main__':
    main()
