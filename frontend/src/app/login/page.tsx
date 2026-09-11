'use client';
import { useState, FormEvent } from 'react';
import { useRouter } from 'next/navigation';
import { apiLogin, apiRegister } from '@/lib/api';

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setLoading(true);
    try {
      if (mode === 'login') {
        const user = await apiLogin(email, password);
        const dest =
          user.role === 'ADMIN'
            ? '/admin'
            : user.role === 'AGENT'
              ? '/agent'
              : '/dashboard';
        router.push(dest);
      } else {
        await apiRegister(name, email, password);
        setSuccess('Account created. You can sign in now.');
        setMode('login');
        setName('');
      }
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
          <div className="auth-tabs" role="tablist" aria-label="Sign in or register">
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
                  setSuccess('');
                }}
              >
                {m === 'login' ? 'Sign in' : 'Register'}
              </button>
            ))}
          </div>

          {error && (
            <div className="alert alert-error mb-2" role="alert">
              {error}
            </div>
          )}
          {success && (
            <div className="alert alert-success mb-2" role="status">
              {success}
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
                <p className="form-hint">At least 8 characters.</p>
              )}
            </div>
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
