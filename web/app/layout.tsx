import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vigil — Command Center",
  description: "Continuous re-triage for waiting rooms: live vision, Claude re-triage, and voice-agent escalation.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="stylesheet" href="/fonts.css" />
      </head>
      <body>{children}</body>
    </html>
  );
}
