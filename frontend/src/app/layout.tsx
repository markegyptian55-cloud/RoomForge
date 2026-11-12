import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = { title: "Room3D", description: "Turn a room video into an explorable 3D scene." };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
