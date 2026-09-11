'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Role, UserInfo, clearToken, getStoredUser } from '@/lib/api';

const ROLE_BADGE: Record<Role, string> = {
  USER: 'badge-user',
  ADMIN: 'badge-admin',
  AGENT: 'badge-agent',
};

const ROLE_LABEL: Record<Role, string> = {
  USER: 'Citizen',
  ADMIN: 'Admin',
  AGENT: 'Agent',
};

export function Navbar() {
  const router = useRouter();
  // localStorage is only available in the browser, so the user is read after
  // mount to keep the server and client markup identical.
  const [user, setUser] = useState<UserInfo | null>(null);

  useEffect(() => {
    setUser(getStoredUser());
  }, []);

  const logout = () => {
    clearToken();
    router.push('/login');
  };

  return (
    <nav className="navbar">
      <span className="navbar-brand">⚡ UrbanEye+</span>
      <div className="navbar-right">
        {user && (
          <>
            <span className={`badge ${ROLE_BADGE[user.role] ?? ''}`}>
              {ROLE_LABEL[user.role] ?? user.role}
            </span>
            <span className="navbar-user">{user.name}</span>
            <button
              className="btn btn-sm btn-outline"
              onClick={logout}
              id="btn-logout"
            >
              Sign out
            </button>
          </>
        )}
      </div>
    </nav>
  );
}
