import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'UrbanEye+ | Smart City Incident Reporting',
  description: 'AI-powered smart city incident reporting and intelligent response system.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
