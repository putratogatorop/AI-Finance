"use client";

import { useEffect, useState } from "react";
import { formatIDR } from "@/lib/format";
import type { RiskSettings } from "@/types";

export default function SettingsPage() {
  const [settings, setSettings] = useState<RiskSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);
  const [formValues, setFormValues] = useState<RiskSettings>({
    stop_loss_pct: -8,
    max_positions: 8,
    confidence_threshold: 0.7,
    position_size_idr: 1_000_000,
    max_single_asset_exposure_pct: 25,
    portfolio_drawdown_pause_pct: 20,
    retrain_schedule: "weekly",
  });

  useEffect(() => {
    async function fetchSettings() {
      setLoading(true);
      try {
        const response = await fetch("/api/settings");
        const result = await response.json();
        if (result.data) {
          setSettings(result.data);
          setFormValues(result.data);
        }
      } catch (error) {
        console.error("Failed to fetch settings:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchSettings();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setSaveMessage(null);

    try {
      const response = await fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formValues),
      });

      const result = await response.json();

      if (response.ok) {
        setSettings(result.data);
        setSaveMessage({ type: "success", text: "Settings saved successfully." });
      } else {
        setSaveMessage({
          type: "error",
          text: result.error || "Failed to save settings.",
        });
      }
    } catch (error) {
      console.error("Failed to save settings:", error);
      setSaveMessage({ type: "error", text: "Network error. Please try again." });
    } finally {
      setSaving(false);
      setTimeout(() => setSaveMessage(null), 5000);
    }
  };

  const handleChange = (field: keyof RiskSettings, value: string | number) => {
    setFormValues((prev) => ({ ...prev, [field]: value }));
  };

  const hasChanges =
    settings !== null &&
    JSON.stringify(settings) !== JSON.stringify(formValues);

  if (loading) {
    return (
      <div className="max-w-2xl space-y-6">
        <div className="card animate-pulse">
          <div className="space-y-4">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-12 rounded bg-slate-700" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-2xl space-y-6">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Risk Management */}
        <div className="card">
          <h2 className="mb-6 text-lg font-semibold text-slate-100">
            Risk Management
          </h2>
          <div className="space-y-5">
            {/* Stop Loss */}
            <SettingsField
              label="Stop Loss (%)"
              description="Hard stop-loss percentage from entry price. Must be negative."
              hint="Current: stop at {value}% loss"
              hintValue={formValues.stop_loss_pct.toString()}
            >
              <input
                type="number"
                value={formValues.stop_loss_pct}
                onChange={(e) =>
                  handleChange("stop_loss_pct", parseFloat(e.target.value))
                }
                min={-50}
                max={0}
                step={0.5}
                className="input-field w-32"
              />
            </SettingsField>

            {/* Max Positions */}
            <SettingsField
              label="Max Concurrent Positions"
              description="Maximum number of open positions at any time."
              hint={`Max exposure: ${formatIDR(formValues.max_positions * formValues.position_size_idr)}`}
            >
              <input
                type="number"
                value={formValues.max_positions}
                onChange={(e) =>
                  handleChange("max_positions", parseInt(e.target.value, 10))
                }
                min={1}
                max={20}
                step={1}
                className="input-field w-32"
              />
            </SettingsField>

            {/* Confidence Threshold */}
            <SettingsField
              label="Confidence Threshold"
              description="Minimum ensemble confidence to generate a signal. Range: 0.1 to 1.0."
              hint={`Signals require ${Math.round(formValues.confidence_threshold * 100)}%+ confidence`}
            >
              <input
                type="number"
                value={formValues.confidence_threshold}
                onChange={(e) =>
                  handleChange(
                    "confidence_threshold",
                    parseFloat(e.target.value)
                  )
                }
                min={0.1}
                max={1.0}
                step={0.05}
                className="input-field w-32"
              />
            </SettingsField>

            {/* Position Size */}
            <SettingsField
              label="Position Size (IDR)"
              description="Fixed amount in IDR per trade entry."
              hint={`Current: ${formatIDR(formValues.position_size_idr)} per trade`}
            >
              <input
                type="number"
                value={formValues.position_size_idr}
                onChange={(e) =>
                  handleChange(
                    "position_size_idr",
                    parseInt(e.target.value, 10)
                  )
                }
                min={100_000}
                max={100_000_000}
                step={100_000}
                className="input-field w-48"
              />
            </SettingsField>

            {/* Max Single Asset Exposure */}
            <SettingsField
              label="Max Single Asset Exposure (%)"
              description="Maximum percentage of total capital in a single asset."
            >
              <input
                type="number"
                value={formValues.max_single_asset_exposure_pct}
                onChange={(e) =>
                  handleChange(
                    "max_single_asset_exposure_pct",
                    parseFloat(e.target.value)
                  )
                }
                min={5}
                max={100}
                step={5}
                className="input-field w-32"
              />
            </SettingsField>

            {/* Portfolio Drawdown Pause */}
            <SettingsField
              label="Portfolio Drawdown Pause (%)"
              description="Pause new signals if total portfolio drops this much from peak."
            >
              <input
                type="number"
                value={formValues.portfolio_drawdown_pause_pct}
                onChange={(e) =>
                  handleChange(
                    "portfolio_drawdown_pause_pct",
                    parseFloat(e.target.value)
                  )
                }
                min={5}
                max={50}
                step={5}
                className="input-field w-32"
              />
            </SettingsField>
          </div>
        </div>

        {/* Schedule */}
        <div className="card">
          <h2 className="mb-6 text-lg font-semibold text-slate-100">
            Model & Data Schedule
          </h2>
          <div className="space-y-5">
            <SettingsField
              label="Retrain Schedule"
              description="How often to retrain ML models with latest data."
            >
              <select
                value={formValues.retrain_schedule}
                onChange={(e) =>
                  handleChange("retrain_schedule", e.target.value)
                }
                className="input-field w-40"
              >
                <option value="daily">Daily</option>
                <option value="weekly">Weekly (Sunday)</option>
                <option value="biweekly">Bi-weekly</option>
                <option value="monthly">Monthly</option>
              </select>
            </SettingsField>

            <div className="rounded-lg border border-slate-700 bg-slate-800/50 p-4">
              <h3 className="text-sm font-medium text-slate-300">
                Fixed Schedules
              </h3>
              <ul className="mt-2 space-y-1 text-xs text-slate-400">
                <li>
                  <span className="text-slate-300">Hourly:</span> Fetch latest candle for all 20 assets, run ETL
                </li>
                <li>
                  <span className="text-slate-300">Daily (00:00 UTC):</span> Aggregate hourly to daily, generate signals, re-evaluate positions
                </li>
                <li>
                  <span className="text-slate-300">Weekly (Sunday):</span> Refresh fundamental data, update asset universe
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Save button */}
        <div className="flex items-center gap-4">
          <button
            type="submit"
            disabled={saving || !hasChanges}
            className="btn-primary"
          >
            {saving ? "Saving..." : "Save Settings"}
          </button>
          {hasChanges && (
            <button
              type="button"
              onClick={() => settings && setFormValues(settings)}
              className="btn-secondary"
            >
              Reset
            </button>
          )}
          {saveMessage && (
            <span
              className={`text-sm ${
                saveMessage.type === "success"
                  ? "text-green-400"
                  : "text-red-400"
              }`}
            >
              {saveMessage.text}
            </span>
          )}
        </div>
      </form>
    </div>
  );
}

function SettingsField({
  label,
  description,
  hint,
  hintValue,
  children,
}: {
  label: string;
  description: string;
  hint?: string;
  hintValue?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
      <div className="flex-1">
        <label className="block text-sm font-medium text-slate-200">
          {label}
        </label>
        <p className="mt-0.5 text-xs text-slate-500">{description}</p>
        {hint && (
          <p className="mt-0.5 text-xs text-slate-400">
            {hintValue ? hint.replace("{value}", hintValue) : hint}
          </p>
        )}
      </div>
      <div className="sm:ml-4">{children}</div>
    </div>
  );
}
