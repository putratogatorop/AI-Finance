import {
  formatIDR,
  formatIDRCompact,
  formatPercent,
  formatUSD,
  formatConfidence,
  getUrgencyLevel,
  getPnlColor,
  getUrgencyBadgeClass,
} from "@/lib/format";

describe("formatIDR", () => {
  it("formats positive amounts", () => {
    expect(formatIDR(1500000)).toBe("Rp 1.500.000");
    expect(formatIDR(1000000)).toBe("Rp 1.000.000");
    expect(formatIDR(0)).toBe("Rp 0");
    expect(formatIDR(1234567890)).toBe("Rp 1.234.567.890");
  });

  it("formats negative amounts", () => {
    expect(formatIDR(-250000)).toBe("-Rp 250.000");
    expect(formatIDR(-1500000)).toBe("-Rp 1.500.000");
  });

  it("rounds to nearest integer", () => {
    expect(formatIDR(1500000.7)).toBe("Rp 1.500.001");
    expect(formatIDR(1500000.3)).toBe("Rp 1.500.000");
  });
});

describe("formatIDRCompact", () => {
  it("formats millions as jt", () => {
    expect(formatIDRCompact(1500000)).toBe("Rp 1,5 jt");
    expect(formatIDRCompact(8000000)).toBe("Rp 8,0 jt");
  });

  it("formats billions as M", () => {
    expect(formatIDRCompact(1500000000)).toBe("Rp 1,5 M");
  });

  it("falls back to full format for small values", () => {
    expect(formatIDRCompact(250000)).toBe("Rp 250.000");
  });

  it("handles negative compact values", () => {
    expect(formatIDRCompact(-2500000)).toBe("-Rp 2,5 jt");
  });
});

describe("formatPercent", () => {
  it("adds plus sign for positive values", () => {
    expect(formatPercent(12.5)).toBe("+12.50%");
  });

  it("shows minus sign for negative values", () => {
    expect(formatPercent(-8.3)).toBe("-8.30%");
  });

  it("shows zero without sign", () => {
    expect(formatPercent(0)).toBe("0.00%");
  });

  it("respects custom decimal places", () => {
    expect(formatPercent(12.5, 1)).toBe("+12.5%");
  });
});

describe("formatUSD", () => {
  it("formats standard prices", () => {
    expect(formatUSD(70123.45)).toBe("$70,123.45");
  });

  it("formats small prices with precision", () => {
    expect(formatUSD(0.0034)).toBe("$0.003400");
  });
});

describe("formatConfidence", () => {
  it("converts decimal to percentage", () => {
    expect(formatConfidence(0.84)).toBe("84%");
    expect(formatConfidence(0.7)).toBe("70%");
    expect(formatConfidence(1.0)).toBe("100%");
  });
});

describe("getUrgencyLevel", () => {
  it("returns red for EXIT action", () => {
    expect(
      getUrgencyLevel({ action: "EXIT", confidence: 0.9, stop_loss_pct: -8 })
    ).toBe("red");
  });

  it("returns red for SELL action", () => {
    expect(
      getUrgencyLevel({ action: "SELL", confidence: 0.8, stop_loss_pct: -8 })
    ).toBe("red");
  });

  it("returns yellow for low confidence BUY", () => {
    expect(
      getUrgencyLevel({ action: "BUY", confidence: 0.55, stop_loss_pct: -8 })
    ).toBe("yellow");
  });

  it("returns green for high confidence BUY", () => {
    expect(
      getUrgencyLevel({ action: "BUY", confidence: 0.84, stop_loss_pct: -8 })
    ).toBe("green");
  });
});

describe("getPnlColor", () => {
  it("returns green for positive P&L", () => {
    expect(getPnlColor(100000)).toBe("text-green-400");
  });

  it("returns red for negative P&L", () => {
    expect(getPnlColor(-50000)).toBe("text-red-400");
  });

  it("returns slate for zero P&L", () => {
    expect(getPnlColor(0)).toBe("text-slate-400");
  });
});

describe("getUrgencyBadgeClass", () => {
  it("returns correct badge classes", () => {
    expect(getUrgencyBadgeClass("green")).toBe("badge-green");
    expect(getUrgencyBadgeClass("yellow")).toBe("badge-yellow");
    expect(getUrgencyBadgeClass("red")).toBe("badge-red");
  });
});
