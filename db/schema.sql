-- 每日特征快照（L1 输出，回测核心表）
CREATE TABLE IF NOT EXISTS feature_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code TEXT NOT NULL,
    as_of TEXT NOT NULL,
    composite_score REAL,
    sector_score REAL,
    capital_score REAL,
    catalyst_score REAL,
    structure_score REAL,
    liquidity_score REAL,
    sector_trace TEXT,
    capital_trace TEXT,
    catalyst_trace TEXT,
    structure_trace TEXT,
    liquidity_trace TEXT,
    feature_vector TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- 买入信号记录
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    mode TEXT DEFAULT 'live',
    sentiment_score REAL,
    sentiment_sector TEXT,
    sentiment_phase TEXT,
    sentiment_detail TEXT,
    composite_score REAL,
    entry_price REAL,
    target_price REAL,
    stop_price REAL,
    rr_ratio REAL,
    position_size REAL,
    primary_driver TEXT,
    core_reason TEXT,
    decision_trace TEXT,
    rationale_text TEXT,
    status TEXT DEFAULT 'ACTIVE',
    created_at TEXT DEFAULT (datetime('now'))
);

-- 持仓记录
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code TEXT NOT NULL,
    signal_id INTEGER REFERENCES signals(id),
    entry_date TEXT,
    entry_price REAL,
    current_price REAL,
    target_price REAL,
    stop_price REAL,
    position_size REAL,
    holding_days INTEGER DEFAULT 0,
    pnl_pct REAL DEFAULT 0,
    status TEXT DEFAULT 'OPEN',
    exit_date TEXT,
    exit_price REAL,
    exit_trigger TEXT,
    updated_at TEXT DEFAULT (datetime('now'))
);

-- 每日持仓评估记录（position_agent 输出）
CREATE TABLE IF NOT EXISTS position_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER REFERENCES positions(id),
    review_date TEXT,
    current_score REAL,
    score_trend TEXT,
    action TEXT,
    size_change REAL DEFAULT 0,
    reason TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- L3 复盘记录
CREATE TABLE IF NOT EXISTS postmortems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id INTEGER REFERENCES positions(id),
    trigger_reason TEXT,
    advocate_output TEXT,
    challenger_output TEXT,
    arbitrator_verdict TEXT,
    arbitrator_confidence REAL,
    execution_class TEXT,
    param_change TEXT,
    human_queue_item TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- 参数变更日志（audit trail）
CREATE TABLE IF NOT EXISTS param_changelog (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL,
    changed_at TEXT,
    agent_name TEXT,
    param_name TEXT,
    value_from TEXT,
    value_to TEXT,
    llm_reasoning TEXT,
    execution_class TEXT,
    falsifiable_expectation TEXT,
    rollback_condition TEXT,
    rollback_date TEXT,
    actual_outcome TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- LLM token 用量与成本记录
CREATE TABLE IF NOT EXISTS llm_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    component TEXT DEFAULT '',
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    estimated_cost_cny REAL DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    status TEXT DEFAULT 'ok',
    error_message TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Migration registry
CREATE TABLE IF NOT EXISTS schema_migrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    migration_id TEXT NOT NULL UNIQUE,
    applied_at TEXT DEFAULT (datetime('now')),
    notes TEXT
);

-- 宏观开关日志（用于日报与运行追溯）
CREATE TABLE IF NOT EXISTS macro_switch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    check_date TEXT NOT NULL,
    result TEXT NOT NULL,
    csi300_close REAL,
    ma60 REAL,
    regime_state TEXT DEFAULT 'RUN',
    regime_evidence_json TEXT,
    regime_policy_json TEXT,
    source TEXT DEFAULT 'scheduler',
    created_at TEXT DEFAULT (datetime('now'))
);

-- buy_agent 决策审计记录（包含 BUY/PASS）
CREATE TABLE IF NOT EXISTS buy_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_date TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    decision TEXT NOT NULL,
    composite_score REAL,
    reason TEXT,
    pass_reason TEXT,
    model TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- 动态权重分配日志
CREATE TABLE IF NOT EXISTS weight_allocation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    weights_json TEXT NOT NULL,
    reasoning TEXT,
    market_summary_json TEXT,
    agent_summaries_json TEXT,
    fallback_used INTEGER DEFAULT 0,
    llm_model TEXT,
    llm_latency_ms INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

-- 候选决策统一审计链（单一事实源）
CREATE TABLE IF NOT EXISTS candidate_decision_trace (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'live',
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    reason_code TEXT,
    details_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Prescreener run-level structured diagnostics
CREATE TABLE IF NOT EXISTS prescreener_run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    trade_date TEXT NOT NULL,
    price_date TEXT NOT NULL,
    candidate_source_mode TEXT NOT NULL,
    prescreener_version TEXT NOT NULL,
    sector_scoring_version TEXT NOT NULL,
    theme_score_version TEXT DEFAULT '',
    review_rank_version TEXT DEFAULT '',
    regime_state TEXT DEFAULT 'RUN',
    market_count INTEGER DEFAULT 0,
    layer1_count INTEGER DEFAULT 0,
    top_n_count INTEGER DEFAULT 0,
    selected_for_opus_count INTEGER DEFAULT 0,
    etf_list_count INTEGER DEFAULT 0,
    etf_daily_rows INTEGER DEFAULT 0,
    funnel_json TEXT,
    top_sectors_json TEXT,
    top_themes_by_total_json TEXT,
    top_themes_by_trend_json TEXT,
    promoted_wildcard_themes_json TEXT,
    excluded_by_theme_cap_json TEXT,
    selected_sectors_json TEXT,
    tech_monitor_json TEXT,
    notes_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS prescreener_theme (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    price_date TEXT NOT NULL,
    theme_name TEXT NOT NULL,
    theme_score_version TEXT NOT NULL,
    theme_total_score REAL DEFAULT 0,
    theme_trend_score REAL DEFAULT 0,
    theme_sentiment_score REAL DEFAULT 0,
    theme_rank_total INTEGER DEFAULT 0,
    theme_rank_trend INTEGER DEFAULT 0,
    quota_allocated INTEGER DEFAULT 0,
    wildcard_promoted INTEGER DEFAULT 0,
    excluded_due_to_theme_cap INTEGER DEFAULT 0,
    quota_reason TEXT DEFAULT '',
    candidate_count INTEGER DEFAULT 0,
    member_count INTEGER DEFAULT 0,
    score_components_json TEXT,
    notes_json TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(run_id, theme_name)
);

-- Prescreener candidate-level structured diagnostics
CREATE TABLE IF NOT EXISTS prescreener_candidate (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    price_date TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    rank_position INTEGER DEFAULT 0,
    selected_for_opus INTEGER DEFAULT 0,
    selected_for_opus_rank INTEGER DEFAULT 0,
    candidate_source_mode TEXT NOT NULL,
    prescore REAL DEFAULT 0,
    sentiment_score REAL DEFAULT 0,
    trend_hotness_score REAL DEFAULT 0,
    stock_quality_score REAL DEFAULT 0,
    role_strength_score REAL DEFAULT 0,
    review_rank_score REAL DEFAULT 0,
    selected_sector_primary TEXT,
    selected_sector_secondary TEXT,
    sector_selection_reason TEXT,
    phase TEXT,
    etf_code TEXT,
    all_matched_sectors_json TEXT,
    matched_sectors_topk_json TEXT,
    sector_scores_by_name_json TEXT,
    score_components_json TEXT,
    review_rank_components_json TEXT,
    selection_bucket TEXT DEFAULT '',
    notes_json TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(run_id, stock_code)
);

CREATE TABLE IF NOT EXISTS prescreener_replacement (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    price_date TEXT NOT NULL,
    rank_slot INTEGER DEFAULT 0,
    replaced_out_stock TEXT NOT NULL,
    promoted_in_stock TEXT NOT NULL,
    old_rank_score REAL DEFAULT 0,
    new_rank_score REAL DEFAULT 0,
    sector_total_diff REAL DEFAULT 0,
    stock_quality_diff REAL DEFAULT 0,
    role_strength_diff REAL DEFAULT 0,
    replacement_reason TEXT NOT NULL,
    details_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Per-task/date lock
CREATE TABLE IF NOT EXISTS task_locks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_name TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    owner TEXT NOT NULL,
    acquired_at TEXT DEFAULT (datetime('now')),
    expires_at TEXT,
    UNIQUE(job_name, trade_date)
);

-- Job run registry
CREATE TABLE IF NOT EXISTS job_run_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    job_name TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    trigger_type TEXT NOT NULL,
    start_at TEXT DEFAULT (datetime('now')),
    end_at TEXT,
    status TEXT NOT NULL,
    retry_count INTEGER DEFAULT 0,
    is_rerun INTEGER DEFAULT 0,
    supersedes_run_id TEXT,
    error_message TEXT,
    strategy_versions_json TEXT,
    report_schema_version TEXT,
    execution_policy_version TEXT
);

-- Strategy version registry
CREATE TABLE IF NOT EXISTS strategy_version_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    version_value TEXT NOT NULL,
    version_hash TEXT NOT NULL,
    experiment_tag TEXT DEFAULT 'baseline',
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(key, experiment_tag)
);

CREATE TABLE IF NOT EXISTS strategy_validation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    validation_id TEXT NOT NULL UNIQUE,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    baseline_frozen_version TEXT NOT NULL,
    candidate_frozen_version TEXT NOT NULL,
    split_registry_json TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    overfit_risk TEXT DEFAULT '',
    promotion_decision TEXT DEFAULT '',
    complexity_score REAL DEFAULT 0,
    complexity_delta_vs_baseline REAL DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS strategy_ablation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    validation_id TEXT NOT NULL,
    rule_name TEXT NOT NULL,
    rule_type TEXT DEFAULT '',
    design_delta REAL DEFAULT 0,
    validation_delta REAL DEFAULT 0,
    holdout_delta REAL DEFAULT 0,
    complexity_delta REAL DEFAULT 0,
    classification TEXT DEFAULT '',
    details_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Weekly audit findings
CREATE TABLE IF NOT EXISTS audit_findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    week_end TEXT NOT NULL,
    category TEXT NOT NULL,
    severity TEXT NOT NULL,
    finding_text TEXT NOT NULL,
    evidence_json TEXT,
    suggested_action TEXT,
    status TEXT DEFAULT 'open',
    source_run_id TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Local metrics sink
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    run_id TEXT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    tags_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Reflection bundles (compressed evidence panels for model reflection)
CREATE TABLE IF NOT EXISTS reflection_bundles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bundle_id TEXT NOT NULL UNIQUE,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    window_days INTEGER NOT NULL,
    strategy_versions_json TEXT,
    bundle_json TEXT NOT NULL,
    sample_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Three-node model review chain outputs
CREATE TABLE IF NOT EXISTS model_review_chain (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chain_id TEXT NOT NULL,
    bundle_id TEXT NOT NULL,
    role TEXT NOT NULL,
    model_name TEXT,
    output_json TEXT NOT NULL,
    status TEXT DEFAULT 'ok',
    created_at TEXT DEFAULT (datetime('now'))
);

-- Governance proposal objects (models can propose, cannot auto-promote)
CREATE TABLE IF NOT EXISTS upgrade_proposals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id TEXT NOT NULL UNIQUE,
    created_at TEXT DEFAULT (datetime('now')),
    created_from_run_window TEXT NOT NULL,
    created_from_audit_id TEXT,
    proposer_model TEXT NOT NULL,
    target_component TEXT NOT NULL,
    current_version TEXT,
    proposed_change_type TEXT NOT NULL,
    proposed_patch_summary TEXT NOT NULL,
    rationale TEXT NOT NULL,
    evidence_refs_json TEXT NOT NULL,
    expected_benefit TEXT,
    primary_risk TEXT,
    rollback_condition TEXT,
    approval_status TEXT NOT NULL DEFAULT 'draft',
    experiment_plan_json TEXT
);

-- Experiment registry for proposal validation
CREATE TABLE IF NOT EXISTS governance_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL UNIQUE,
    experiment_tag TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    base_version TEXT,
    candidate_version TEXT,
    start_date TEXT NOT NULL,
    end_date TEXT,
    success_criteria_json TEXT NOT NULL,
    stop_conditions_json TEXT NOT NULL,
    verdict TEXT DEFAULT 'inconclusive',
    verdict_note TEXT,
    status TEXT DEFAULT 'running',
    created_at TEXT DEFAULT (datetime('now'))
);

-- Human approval registry
CREATE TABLE IF NOT EXISTS governance_approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL, -- proposal|experiment|promotion
    entity_id TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    decision TEXT NOT NULL, -- approve|reject|hold
    decision_time TEXT DEFAULT (datetime('now')),
    decision_note TEXT,
    approved_scope TEXT,
    rollback_plan_ack INTEGER DEFAULT 0,
    UNIQUE(entity_type, entity_id, reviewer)
);

-- Live promotion lineage (must reference proposal + experiment)
CREATE TABLE IF NOT EXISTS live_promotion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    promotion_id TEXT NOT NULL UNIQUE,
    proposal_id TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    promoted_version TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    decision_note TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Regret evaluation records
CREATE TABLE IF NOT EXISTS regret_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'live',
    final_outcome TEXT NOT NULL,
    regret_type TEXT NOT NULL,
    severity_score REAL DEFAULT 0,
    observation_window TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    evidence_refs_json TEXT NOT NULL,
    taxonomy_version TEXT NOT NULL,
    sample_window TEXT NOT NULL,
    version_snapshot_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Research prioritization output
CREATE TABLE IF NOT EXISTS research_priorities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of_date TEXT NOT NULL,
    priority_level TEXT NOT NULL,
    problem_statement TEXT NOT NULL,
    suspected_root_cause TEXT,
    likely_target_component TEXT,
    recommended_change_type TEXT,
    supporting_evidence_json TEXT,
    score_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Offline replay harness results
CREATE TABLE IF NOT EXISTS replay_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    replay_id TEXT NOT NULL UNIQUE,
    sample_window TEXT NOT NULL,
    component_type TEXT NOT NULL,
    base_version TEXT NOT NULL,
    candidate_version TEXT NOT NULL,
    diff_summary_json TEXT NOT NULL,
    limitation_note TEXT,
    version_snapshot_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Research memo knowledge objects
CREATE TABLE IF NOT EXISTS research_memos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memo_id TEXT NOT NULL UNIQUE,
    hypothesis TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    verdict TEXT NOT NULL,
    expiry_condition TEXT,
    status TEXT DEFAULT 'active',
    created_at TEXT DEFAULT (datetime('now'))
);

-- Opus 决策日志（RUN 模式主决策审计）
CREATE TABLE IF NOT EXISTS opus_decision_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    decisions_json TEXT NOT NULL,
    market_assessment TEXT,
    adversarial_summary TEXT,
    overrides_count INTEGER DEFAULT 0,
    tool_calls_json TEXT,
    conversation_json TEXT,
    total_rounds INTEGER,
    total_input_tokens INTEGER,
    total_output_tokens INTEGER,
    latency_seconds REAL,
    status TEXT DEFAULT 'success',
    error_message TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_feature_snapshots_stock_date ON feature_snapshots(stock_code, as_of);
CREATE INDEX IF NOT EXISTS idx_signals_stock_date ON signals(stock_code, signal_date);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);
CREATE INDEX IF NOT EXISTS idx_signals_mode_date ON signals(mode, signal_date);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_stock ON positions(stock_code);
CREATE INDEX IF NOT EXISTS idx_postmortems_created ON postmortems(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_created ON llm_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_component ON llm_usage(component, created_at);
CREATE INDEX IF NOT EXISTS idx_macro_switch_date ON macro_switch_log(check_date);
CREATE INDEX IF NOT EXISTS idx_buy_decisions_date ON buy_decisions(decision_date);
CREATE INDEX IF NOT EXISTS idx_buy_decisions_decision ON buy_decisions(decision, decision_date);
CREATE INDEX IF NOT EXISTS idx_weight_allocation_date ON weight_allocation_log(trade_date);
CREATE INDEX IF NOT EXISTS idx_trace_date_mode_stock ON candidate_decision_trace(trade_date, mode, stock_code);
CREATE INDEX IF NOT EXISTS idx_trace_stage ON candidate_decision_trace(trade_date, mode, stage, status);
CREATE INDEX IF NOT EXISTS idx_prescreener_run_lookup ON prescreener_run(trade_date, run_id);
CREATE INDEX IF NOT EXISTS idx_prescreener_candidate_lookup ON prescreener_candidate(trade_date, run_id, rank_position);
CREATE INDEX IF NOT EXISTS idx_prescreener_theme_lookup ON prescreener_theme(trade_date, run_id, theme_rank_total);
CREATE INDEX IF NOT EXISTS idx_opus_decision_date ON opus_decision_log(trade_date);
CREATE INDEX IF NOT EXISTS idx_job_run_lookup ON job_run_log(job_name, trade_date, status, id);
CREATE INDEX IF NOT EXISTS idx_system_metrics_lookup ON system_metrics(trade_date, metric_name);
CREATE INDEX IF NOT EXISTS idx_audit_findings_week ON audit_findings(week_end, severity);
CREATE INDEX IF NOT EXISTS idx_reflection_bundle_window ON reflection_bundles(window_start, window_end);
CREATE INDEX IF NOT EXISTS idx_model_review_chain_chain ON model_review_chain(chain_id, role);
CREATE INDEX IF NOT EXISTS idx_upgrade_proposals_status ON upgrade_proposals(approval_status, proposed_change_type);
CREATE INDEX IF NOT EXISTS idx_governance_experiments_tag ON governance_experiments(experiment_tag, status);
CREATE INDEX IF NOT EXISTS idx_governance_approvals_entity ON governance_approvals(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_regret_records_date_type ON regret_records(trade_date, regret_type, mode);
CREATE INDEX IF NOT EXISTS idx_research_priorities_asof ON research_priorities(as_of_date, priority_level);
CREATE INDEX IF NOT EXISTS idx_replay_runs_window ON replay_runs(sample_window, component_type);
CREATE INDEX IF NOT EXISTS idx_research_memos_status ON research_memos(status, created_at);
