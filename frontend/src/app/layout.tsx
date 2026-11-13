import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = { title: "RoomForge", description: "Turn a room video into an explorable 3D room." };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en" className="dark"><body>{children}</body></html>; }
