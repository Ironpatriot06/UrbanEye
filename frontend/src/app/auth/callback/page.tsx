'use client';
/**
 * Landing page for the Google sign-in redirect.
 *
 * The backend completes the OAuth exchange, verifies Google's ID token, and
 * sends the browser here with a session token in the query string. This page
 * adopts that session and forwards to the right dashboard.
 *
 * The `role` in the URL decides only which dashboard to open. It is not an
 * authorization decision: every protected request is authorized by the backend
 * from the token and the user's database row, so tampering with it just lands
 * the user on a page that refuses to load.
 */
import { Suspense, useEffect, useRef, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { adoptSession, type Role } from '@/lib/api';

function isRole(value: string | null): value is Role {
  return value === 'USER' || value === 'AGENT' || value === 'ADMIN';
}

function CallbackHandler() {
  const router = useRouter();
  const params = useSearchParams();
  const [error, setError] = useState('');
  // React runs effects twice in development; the redirect must fire once.
  const handled = useRef(false);

  useEffect(() => {
    if (handled.current) return;
    handled.current = true;

    const authError = params.get('auth_error');
    if (authError) {
      setError(authError);
      return;
    }

    const token = params.get('token');
    const role = params.get('role');
    const userId = params.get('user_id');
    const name = params.get('name');

    if (!token || !isRole(role) || !userId) {
      setError('Sign-in did not complete. Please try again from the login page.');
      return;
    }

    adoptSession({
      access_token: token,
      role,
      user_id: Number(userId),
      name: name || 'UrbanEye user',
    });

    // replace, not push: the URL carries a token and should not be somewhere
    // the back button returns to.
    router.replace(role === 'ADMIN' ? '/admin' : role === 'AGENT' ? '/agent' : '/dashboard');
  }, [params, router]);

  if (error) {
    return (
      <main className="auth-shell">
        <div className="auth-card">
          <div className="auth-brand">
            <div className="auth-brand__mark" aria-hidden="true">⚡</div>
            <h1>UrbanEye+</h1>
          </div>
          <div className="card">
            <div className="alert alert-error mb-2" role="alert">
              {error}
            </div>
            <button
              className="btn btn-primary btn-full"
              onClick={() => router.replace('/login')}
            >
              Back to sign in
            </button>
          </div>
        </div>
      </main>
    );
  }

  return (
    <div className="loading-block" style={{ minHeight: '100vh', justifyContent: 'center' }}>
      <span className="spinner" />
      <p>Signing you in…</p>
    </div>
  );
}

export default function GoogleCallbackPage() {
  return (
    <Suspense
      fallback={
        <div className="loading-block" style={{ minHeight: '100vh', justifyContent: 'center' }}>
          <span className="spinner" />
        </div>
      }
    >
      <CallbackHandler />
    </Suspense>
  );
}
