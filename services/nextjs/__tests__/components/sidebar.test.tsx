import { render, screen } from "@testing-library/react";
import { Sidebar } from "@/components/layout/sidebar";

jest.mock("next/navigation", () => ({
  usePathname: () => "/",
}));

describe("Sidebar", () => {
  it("renders all navigation items", () => {
    render(<Sidebar />);
    expect(screen.getAllByText("Overview").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Signals").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Portfolio").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Backtest").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Audit Log").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Settings").length).toBeGreaterThan(0);
  });

  it("renders the brand name", () => {
    render(<Sidebar />);
    expect(screen.getByText("AI-Finance")).toBeInTheDocument();
  });

  it("highlights the active route", () => {
    render(<Sidebar />);
    const overviewLinks = screen.getAllByRole("link", { name: /overview/i });
    const hasActiveClass = overviewLinks.some((link) =>
      link.className.includes("text-brand-400")
    );
    expect(hasActiveClass).toBe(true);
  });
});
