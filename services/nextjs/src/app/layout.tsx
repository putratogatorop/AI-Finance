import type { Metadata } from "next";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI-Finance Dashboard",
  description: "Crypto trading system with ML-driven signals",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full">
        <Sidebar />
        <div className="lg:pl-64">
          <Header />
          <main className="px-4 py-6 sm:px-6 lg:px-8 pb-20 lg:pb-6">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
