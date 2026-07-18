import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://web-pink-chi-71.vercel.app"),
  title: "Vigil | Command Center",
  description: "Live patient monitoring, clinical reasoning, and escalation in one command center.",
  openGraph: {
    title: "Vigil | Command Center",
    description: "Live patient monitoring, clinical reasoning, and escalation in one command center.",
    images: [{ url: "/vigil-preview.png", width: 1200, height: 630, alt: "Vigil clinical command center" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Vigil | Command Center",
    description: "Live patient monitoring, clinical reasoning, and escalation in one command center.",
    images: ["/vigil-preview.png"],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
