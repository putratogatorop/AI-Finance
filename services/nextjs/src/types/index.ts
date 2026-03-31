// ----- Signal types -----

export type SignalAction = "BUY" | "SELL" | "HOLD" | "EXIT";

export type UrgencyLevel = "green" | "yellow" | "red";

export type AcknowledgeStatus = "seen" | "acted" | "skipped";

export interface SignalRecord {
  id: number;
  asset: string;
  action: SignalAction;
  confidence: number;
  suggested_hold_days: number;
  stop_loss_pct: number;
  expected_return_pct: number;
  model_agreement: string;
  acknowledged: boolean;
  created_at: string;
  urgency?: UrgencyLevel;
}

// ----- Portfolio types -----

export type PositionStatus = "open" | "closed" | "stopped";

export interface PortfolioPosition {
  id: number;
  asset: string;
  action: string;
  entry_price: number;
  entry_amount_idr: number;
  quantity: number;
  stop_loss_price: number;
  exit_price: number | null;
  exit_amount_idr: number | null;
  pnl_idr: number | null;
  pnl_pct: number | null;
  status: PositionStatus;
  opened_at: string;
  closed_at: string | null;
  signal_id: number | null;
  current_price?: number;
  unrealized_pnl_idr?: number;
  unrealized_pnl_pct?: number;
}

// ----- Price types -----

export interface PricePoint {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface DailyPrice {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// ----- Backtest types -----

export interface BacktestMetrics {
  total_return_idr: number;
  total_return_pct: number;
  win_rate: number;
  reward_risk_ratio: number;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
}

export interface MonthlyBreakdown {
  month: string;
  return_idr: number;
  return_pct: number;
  trades: number;
  win_rate: number;
}

export interface BacktestResult {
  metrics: BacktestMetrics;
  monthly: MonthlyBreakdown[];
  equity_curve: { date: string; value: number }[];
}

// ----- Asset detail types -----

export interface AssetDetail {
  symbol: string;
  current_price: number;
  price_history: DailyPrice[];
  fundamentals: {
    market_cap: number | null;
    market_cap_rank: number | null;
    total_volume_24h: number | null;
    circulating_supply: number | null;
    category: string | null;
  } | null;
  recent_signals: SignalRecord[];
  trade_history: PortfolioPosition[];
  model_scores: {
    xgboost: number | null;
    lightgbm: number | null;
    lstm: number | null;
    ensemble: number | null;
  };
  features: Record<string, number>;
}

// ----- Audit types -----

export interface AuditEntry {
  id: number;
  event_type: string;
  asset: string | null;
  details: string | null;
  created_at: string;
  parsed_details?: Record<string, unknown>;
}

// ----- Settings types -----

export interface RiskSettings {
  stop_loss_pct: number;
  max_positions: number;
  confidence_threshold: number;
  position_size_idr: number;
  max_single_asset_exposure_pct: number;
  portfolio_drawdown_pause_pct: number;
  retrain_schedule: string;
}

// ----- Overview types -----

export interface OverviewData {
  portfolio_value_idr: number;
  total_pnl_idr: number;
  total_pnl_pct: number;
  open_positions_count: number;
  max_positions: number;
  todays_signals: SignalRecord[];
  portfolio_history: { date: string; value: number }[];
}

// ----- API response wrapper -----

export interface ApiResponse<T> {
  data: T;
  error?: string;
}
