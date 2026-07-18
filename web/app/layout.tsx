import type { Metadata } from "next";
import "./globals.css";

const siteUrl = process.env.NEXT_PUBLIC_SITE_URL || "https://web-pink-chi-71.vercel.app";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: "Vigil | Command Center",
  description: "Continuous waiting-room re-triage with role-scoped response and audit tools.",
  openGraph: {
    title: "Vigil | Command Center",
    description: "Continuous waiting-room re-triage with role-scoped response and audit tools.",
  },
  twitter: {
    card: "summary",
    title: "Vigil | Command Center",
    description: "Continuous waiting-room re-triage with role-scoped response and audit tools.",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
