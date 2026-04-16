import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "West Africa Microfinance Credit Risk Scorecard",
  description:
    "Basel II-compliant credit scorecard for West African microfinance. WoE/IV feature selection, logistic regression with points conversion, Gini/KS/PSI validation, and stress testing.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#0a0a0a] font-sans text-neutral-200 antialiased">
        {children}
      </body>
    </html>
  );
}
