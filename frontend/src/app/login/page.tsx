'use client';
import { Suspense, useEffect, useState, FormEvent } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import {
  apiAuthConfig,
  apiLogin,
  apiRegister,
  startGoogleSignIn,
  type Role,
} from '@/lib/api';

/** Where each role lands after signing in. */
function landingFor(role: Role): string {
  return role === 'ADMIN' ? '/admin' : role === 'AGENT' ? '/agent' : '/dashboard';
}

function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();

  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Hidden unless the server actually has Google credentials configured, so
  // the button is never offered when it could only fail.
  const [googleEnabled, setGoogleEnabled] = useState(false);

  useEffect(() => {
    apiAuthConfig()
      .then((cfg) => setGoogleEnabled(cfg.google_enabled))
      .catch(() => setGoogleEnabled(false));
  }, []);

  // The Google callback redirects here with ?auth_error=... when sign-in could
  // not be completed or verified.
  useEffect(() => {
    const authError = params.get('auth_error');
    if (authError) setError(authError);
  }, [params]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');

    if (mode === 'register' && password !== confirm) {
      setError('The two passwords do not match.');
      return;
    }

    setLoading(true);
    try {
      // Registration signs the new account in directly — the role is always
      // USER, decided by the backend, so the destination is the citizen
      // dashboard.
      const user =
        mode === 'login'
          ? await apiLogin(email, password)
          : await apiRegister(name, email, password);
      router.push(landingFor(user.role));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="auth-shell">
      <div className="auth-card">
        <div className="auth-brand">
          <div className="auth-brand__mark" aria-hidden="true">⚡</div>
          <h1>UrbanEye+</h1>
          <p>Smart city incident reporting</p>
        </div>

        <div className="card">
          <div className="auth-tabs" role="tablist" aria-label="Sign in or create an account">
            {(['login', 'register'] as const).map((m) => (
              <button
                key={m}
                type="button"
                role="tab"
                aria-selected={mode === m}
                className={`auth-tab${mode === m ? ' is-active' : ''}`}
                onClick={() => {
                  setMode(m);
                  setError('');
                }}
              >
                {m === 'login' ? 'Sign in' : 'Create account'}
              </button>
            ))}
          </div>

          {error && (
            <div className="alert alert-error mb-2" role="alert">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="form-stack">
            {mode === 'register' && (
              <div className="form-group">
                <label className="form-label" htmlFor="input-name">
                  Full name
                </label>
                <input
                  id="input-name"
                  className="form-input"
                  type="text"
                  autoComplete="name"
                  placeholder="Riya Sharma"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                  maxLength={100}
                />
              </div>
            )}

            <div className="form-group">
              <label className="form-label" htmlFor="input-email">
                Email
              </label>
              <input
                id="input-email"
                className="form-input"
                type="email"
                autoComplete="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="input-password">
                Password
              </label>
              <input
                id="input-password"
                className="form-input"
                type="password"
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={8}
              />
              {mode === 'register' && (
                <p className="form-hint">
                  At least 8 characters, including a letter and a number.
                </p>
              )}
            </div>

            {mode === 'register' && (
              <div className="form-group">
                <label className="form-label" htmlFor="input-confirm">
                  Confirm password
                </label>
                <input
                  id="input-confirm"
                  className="form-input"
                  type="password"
                  autoComplete="new-password"
                  placeholder="••••••••"
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  required
                  minLength={8}
                  aria-invalid={confirm.length > 0 && confirm !== password}
                />
                {confirm.length > 0 && confirm !== password && (
                  <p className="form-hint form-hint--error">
                    The two passwords do not match.
                  </p>
                )}
              </div>
            )}

            <button
              id="btn-submit"
              type="submit"
              className="btn btn-primary btn-full"
              disabled={loading}
            >
              {loading && <span className="spinner spinner-sm" />}
              {loading
                ? 'Please wait…'
                : mode === 'login'
                  ? 'Sign in'
                  : 'Create account'}
            </button>
          </form>

          {googleEnabled && (
            <>
              <div className="auth-divider" role="separator">
                <span>or</span>
              </div>
              <button
                id="btn-google"
                type="button"
                className="btn btn-google btn-full"
                onClick={() => startGoogleSignIn()}
                disabled={loading}
              >
                <GoogleMark />
                Continue with Google
              </button>
              <p className="form-hint auth-note">
                New Google accounts join as citizens. An administrator grants
                agent or admin access afterwards.
              </p>
            </>
          )}

          {mode === 'login' && (
            <div className="demo-accounts">
              <strong>Demo accounts</strong>
              <ul>
                <li>👤 Citizen — <code>user@urbaneye.local</code> / <code>User@1234</code></li>
                <li>🛡 Admin — <code>admin@urbaneye.local</code> / <code>Admin@1234</code></li>
                <li>🔧 Agent — <code>agent@urbaneye.local</code> / <code>Agent@1234</code></li>
              </ul>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}

/** Google's four-colour mark, inline so the page pulls in no external asset. */
function GoogleMark() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true" focusable="false">
      <path
        fill="#4285F4"
        d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.92c1.7-1.57 2.68-3.88 2.68-6.62Z"
      />
      <path
        fill="#34A853"
        d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.92-2.26c-.8.54-1.83.86-3.04.86-2.34 0-4.32-1.58-5.03-3.7H.96v2.33A9 9 0 0 0 9 18Z"
      />
      <path
        fill="#FBBC05"
        d="M3.97 10.72a5.4 5.4 0 0 1 0-3.44V4.95H.96a9 9 0 0 0 0 8.1l3.01-2.33Z"
      />
      <path
        fill="#EA4335"
        d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58C13.46.9 11.43 0 9 0A9 9 0 0 0 .96 4.95l3.01 2.33C4.68 5.16 6.66 3.58 9 3.58Z"
      />
    </svg>
  );
}

export default function LoginPage() {
  // useSearchParams needs a Suspense boundary for the production build.
  return (
    <Suspense
      fallback={
        <div className="loading-block" style={{ minHeight: '100vh', justifyContent: 'center' }}>
          <span className="spinner" />
        </div>
      }
    >
      <LoginForm />
    </Suspense>
  );
}
