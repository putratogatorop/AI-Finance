/**
 * Format a number as Indonesian Rupiah (IDR).
 */
export function formatIDR(amount: number): string {
  const isNegative = amount < 0;
  const absAmount = Math.abs(Math.round(amount));
  const formatted = absAmount.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".");
  return `${isNegative ? "-" : ""}Rp ${formatted}`;
}

/**
 * Format a number as IDR with compact notation for large values.
 */
export function formatIDRCompact(amount: number): string {
  const absAmount = Math.abs(amount);
  const isNegative = amount < 0;
  const sign = isNegative ? "-" : "";

  if (absAmount >= 1_000_000_000) {
    const value = absAmount / 1_000_000_000;
    return `${sign}Rp ${value.toFixed(1).replace(".", ",")} M`;
  }
  if (absAmount >= 1_000_000) {
    const value = absAmount / 1_000_000;
    return `${sign}Rp ${value.toFixed(1).replace(".", ",")} jt`;
  }
  return formatIDR(amount);
}

/**
 * Format a percentage value with sign and fixed decimals.
 */
export function formatPercent(value: number, decimals: number = 2): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(decimals)}%`;
}

/**
 * Format a USD price with appropriate decimal places.
 */
export function formatUSD(amount: number): string {
  if (amount < 0.01 && amount > 0) {
    return `$${amount.toPrecision(4)}`;
  }
  return `$${amount.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

/**
 * Format a confidence score as a percentage.
 */
export function formatConfidence(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/**
 * Format a datetime string to locale display.
 */
export function formatDateTime(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleDateString("id-ID", {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

/**
 * Format a date string to short display.
 */
export function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString("id-ID", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

/**
 * Determine the urgency level for a signal.
 */
export function getUrgencyLevel(signal: {
  action: string;
  confidence: number;
  stop_loss_pct: number;
}): "green" | "yellow" | "red" {
  if (signal.action === "EXIT" || signal.action === "SELL") {
    return "red";
  }
  if (signal.confidence < 0.6) {
    return "yellow";
  }
  if (signal.confidence < 0.5) {
    return "red";
  }
  return "green";
}

/**
 * Get a CSS class name for P&L coloring.
 */
export function getPnlColor(value: number): string {
  if (value > 0) return "text-green-400";
  if (value < 0) return "text-red-400";
  return "text-slate-400";
}

/**
 * Get a CSS class name for urgency badge.
 */
export function getUrgencyBadgeClass(
  urgency: "green" | "yellow" | "red"
): string {
  switch (urgency) {
    case "green":
      return "badge-green";
    case "yellow":
      return "badge-yellow";
    case "red":
      return "badge-red";
  }
}
