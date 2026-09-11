'use client';
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { getStoredUser } from '@/lib/api';

export default function HomePage() {
  const router = useRouter();
  useEffect(() => {
    const user = getStoredUser();
    if (!user) {
      router.replace('/login');
      return;
    }
    const dest =
      user.role === 'ADMIN' ? '/admin' : user.role === 'AGENT' ? '/agent' : '/dashboard';
    router.replace(dest);
  }, [router]);

  return (
    <div className="loading-block" style={{ minHeight: '100vh', justifyContent: 'center' }}>
      <span className="spinner" />
      <p>Taking you to your dashboard…</p>
    </div>
  );
}
