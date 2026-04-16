import { render, screen, fireEvent } from "@testing-library/react";
import { SignalCard } from "@/components/signals/signal-card";
import type { SignalRecord } from "@/types";

const mockSignal: SignalRecord = {
  id: 1,
  asset: "ETH",
  action: "BUY",
  confidence: 0.84,
  suggested_hold_days: 18,
  stop_loss_pct: -8,
  expected_return_pct: 12,
  model_agreement: "3/3",
  acknowledged: false,
  created_at: "2026-03-30T00:15:00Z",
};

describe("SignalCard", () => {
  it("renders signal details", () => {
    render(<SignalCard signal={mockSignal} onAcknowledge={() => {}} />);

    expect(screen.getByText("ETH")).toBeInTheDocument();
    expect(screen.getByText("BUY")).toBeInTheDocument();
    expect(screen.getByText("84%")).toBeInTheDocument();
    expect(screen.getByText("+12.00%")).toBeInTheDocument();
    expect(screen.getByText("18 days")).toBeInTheDocument();
    expect(screen.getByText("3/3")).toBeInTheDocument();
  });

  it("shows acknowledge buttons when not acknowledged", () => {
    render(<SignalCard signal={mockSignal} onAcknowledge={() => {}} />);

    expect(screen.getByText("Act")).toBeInTheDocument();
    expect(screen.getByText("Seen")).toBeInTheDocument();
    expect(screen.getByText("Skip")).toBeInTheDocument();
  });

  it("hides acknowledge buttons when acknowledged", () => {
    const acknowledged = { ...mockSignal, acknowledged: true };
    render(
      <SignalCard signal={acknowledged} onAcknowledge={() => {}} />
    );

    expect(screen.queryByText("Act")).not.toBeInTheDocument();
    expect(screen.getByText("Acknowledged")).toBeInTheDocument();
  });

  it("calls onAcknowledge with correct parameters", () => {
    const onAcknowledge = jest.fn();
    render(
      <SignalCard signal={mockSignal} onAcknowledge={onAcknowledge} />
    );

    fireEvent.click(screen.getByText("Act"));
    expect(onAcknowledge).toHaveBeenCalledWith(1, "acted");

    fireEvent.click(screen.getByText("Seen"));
    expect(onAcknowledge).toHaveBeenCalledWith(1, "seen");

    fireEvent.click(screen.getByText("Skip"));
    expect(onAcknowledge).toHaveBeenCalledWith(1, "skipped");
  });

  it("shows green urgency for high confidence BUY", () => {
    render(<SignalCard signal={mockSignal} onAcknowledge={() => {}} />);
    expect(screen.getByText("GREEN")).toBeInTheDocument();
  });

  it("shows red urgency for SELL action", () => {
    const sell = { ...mockSignal, action: "SELL" as const };
    render(<SignalCard signal={sell} onAcknowledge={() => {}} />);
    expect(screen.getByText("RED")).toBeInTheDocument();
  });

  it("disables buttons when loading", () => {
    render(
      <SignalCard
        signal={mockSignal}
        onAcknowledge={() => {}}
        isLoading
      />
    );

    const buttons = screen.getAllByRole("button");
    buttons.forEach((button) => {
      expect(button).toBeDisabled();
    });
  });
});
