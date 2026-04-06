# New Drawer

This drawer contains the momentum rotation system and the shared modules it still needs.

Primary entry points:
- `backtest/momentum_backtest.py`
- `scheduler/momentum_scheduler.py`
- `scripts/run_tushare_backtest.py`
- `momentum/`
- `evolution/`

Shared dependencies kept here for the new strategy:
- `config/`
- `data_pipeline/tushare_client.py`
- `data_pipeline/clock.py`
- `db/`
- `parameter_versions/`
- `data/`

Notes:
- Top-level `new/` is intended to remain runnable as the active project root.
- Legacy mainline-dragon strategy modules were moved to `../old/`.
