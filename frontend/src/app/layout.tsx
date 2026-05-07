import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Salamander Tracker",
  description: "Upload salamander footage and view YOLO detections.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
