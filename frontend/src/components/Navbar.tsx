'use client';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
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

/**
 * Navigation entries per role.
 *
 * This decides what is SHOWN, never what is permitted. User Management is
 * hidden from citizens and agents as a courtesy — the control is that
 * /api/v1/admin/* answers 403 to them regardless of what the navbar renders or
 * what a hand-edited localStorage claims.
 */
const ROLE_LINKS: Record<Role, { href: string; label: string }[]> = {
  USER: [{ href: '/dashboard', label: 'My reports' }],
  AGENT: [{ href: '/agent', label: 'My queue' }],
  ADMIN: [
    { href: '/admin', label: 'Incidents' },
    { href: '/admin/users', label: 'User Management' },
  ],
};

export function Navbar() {
  const router = useRouter();
  const pathname = usePathname();
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

  const links = user ? ROLE_LINKS[user.role] ?? [] : [];

  return (
    <nav className="navbar">
      <div className="navbar-left">
        <span className="navbar-brand">⚡ UrbanEye+</span>
        {links.length > 1 && (
          <div className="navbar-links">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`navbar-link${pathname === link.href ? ' is-active' : ''}`}
                aria-current={pathname === link.href ? 'page' : undefined}
              >
                {link.label}
              </Link>
            ))}
          </div>
        )}
      </div>
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
