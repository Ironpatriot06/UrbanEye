/* ======================================================
   API Client — wraps fetch with JWT authentication

   Every request (including image bytes) goes out with an
   `Authorization: Bearer <token>` header.  The backend has no
   query-string token support and images are deliberately NOT public,
   so binary endpoints are fetched with the same header and turned
   into object URLs — see `fetchImageObjectUrl` below.

   The backend origin always comes from NEXT_PUBLIC_API_URL; the
   localhost value is only the development default.
   ====================================================== */

export const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ── Token storage ───────────────────────────────────────────
export function getToken(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem('ue_token');
}
export function setToken(token: string) {
  localStorage.setItem('ue_token', token);
}
export function clearToken() {
  localStorage.removeItem('ue_token');
  localStorage.removeItem('ue_user');
}

export function getStoredUser(): UserInfo | null {
  if (typeof window === 'undefined') return null;
  const raw = localStorage.getItem('ue_user');
  if (!raw) return null;
  try {
    return JSON.parse(raw) as UserInfo;
  } catch {
    return null;
  }
}
export function setStoredUser(u: UserInfo) {
  localStorage.setItem('ue_user', JSON.stringify(u));
}

// ── Types ───────────────────────────────────────────────────
export type Role = 'USER' | 'ADMIN' | 'AGENT';

export type IncidentStatus =
  | 'REPORTED'
  | 'TRIAGED'
  | 'ASSIGNED'
  | 'IN_PROGRESS'
  | 'RESOLVED'
  | 'CLOSED';

export type PriorityLevel = 'P1' | 'P2' | 'P3' | 'P4';

export type SlaStatus = 'ON_TRACK' | 'AT_RISK' | 'BREACHED';

/** The incident lifecycle, in order. Mirrors IncidentStatus on the backend. */
export const WORKFLOW: IncidentStatus[] = [
  'REPORTED',
  'TRIAGED',
  'ASSIGNED',
  'IN_PROGRESS',
  'RESOLVED',
  'CLOSED',
];

/**
 * Transitions an agent may perform. Mirrors AGENT_TRANSITIONS in
 * app/models/incident.py — the backend is the authority and rejects
 * anything else with a 403; this map only shapes the UI.
 */
export const AGENT_TRANSITIONS: Partial<Record<IncidentStatus, IncidentStatus[]>> = {
  ASSIGNED: ['IN_PROGRESS'],
  IN_PROGRESS: ['RESOLVED'],
};

/**
 * Transitions an admin may perform. Mirrors VALID_TRANSITIONS in
 * app/models/incident.py. Again: advisory only, enforced server-side.
 */
export const ADMIN_TRANSITIONS: Record<IncidentStatus, IncidentStatus[]> = {
  REPORTED: ['TRIAGED', 'ASSIGNED', 'CLOSED'],
  TRIAGED: ['ASSIGNED', 'CLOSED'],
  ASSIGNED: ['IN_PROGRESS', 'TRIAGED', 'CLOSED'],
  IN_PROGRESS: ['RESOLVED', 'ASSIGNED'],
  RESOLVED: ['CLOSED', 'IN_PROGRESS'],
  CLOSED: [],
};

export const STATUS_LABELS: Record<IncidentStatus, string> = {
  REPORTED: 'Reported',
  TRIAGED: 'Triaged',
  ASSIGNED: 'Assigned',
  IN_PROGRESS: 'In Progress',
  RESOLVED: 'Resolved',
  CLOSED: 'Closed',
};

export const PRIORITY_LABELS: Record<PriorityLevel, string> = {
  P1: 'Critical',
  P2: 'High',
  P3: 'Medium',
  P4: 'Low',
};

export const SLA_LABELS: Record<SlaStatus, string> = {
  ON_TRACK: 'On track',
  AT_RISK: 'At risk',
  BREACHED: 'Breached',
};

export const CATEGORY_LABELS: Record<string, string> = {
  POTHOLE: 'Pothole',
  FLOOD: 'Flood',
  FIRE_HAZARD: 'Fire hazard',
  GARBAGE: 'Garbage',
  STREETLIGHT: 'Streetlight',
  OTHER: 'Other',
};

/**
 * The controlled set of auditable actions. Mirrors HistoryAction in
 * app/models/history.py — the backend is the authority; this exists so the UI
 * can label and group events.
 */
export type HistoryActionType =
  | 'INCIDENT_CREATED'
  | 'INCIDENT_TRIAGED'
  | 'PRIORITY_CHANGED'
  | 'SLA_CREATED'
  | 'SLA_UPDATED'
  | 'AGENT_ASSIGNED'
  | 'AGENT_REASSIGNED'
  | 'AGENT_UNASSIGNED'
  | 'STATUS_CHANGED'
  | 'AGENT_AVAILABILITY_CHANGED'
  | 'IMAGE_ADDED'
  | 'INCIDENT_RESOLVED'
  | 'INCIDENT_CLOSED'
  | 'INCIDENT_REOPENED'
  | 'ADMIN_OVERRIDE';

/** Mirrors ActorRole on the backend: the three user roles plus SYSTEM. */
export type ActorRole = Role | 'SYSTEM';

/** Short headline for each event type, shown as the timeline entry's title. */
export const HISTORY_ACTION_LABELS: Record<HistoryActionType, string> = {
  INCIDENT_CREATED: 'Incident reported',
  INCIDENT_TRIAGED: 'Triaged',
  PRIORITY_CHANGED: 'Priority changed',
  SLA_CREATED: 'Response target set',
  SLA_UPDATED: 'Response target changed',
  AGENT_ASSIGNED: 'Agent assigned',
  AGENT_REASSIGNED: 'Agent reassigned',
  AGENT_UNASSIGNED: 'Agent unassigned',
  STATUS_CHANGED: 'Status changed',
  AGENT_AVAILABILITY_CHANGED: 'Agent availability changed',
  IMAGE_ADDED: 'Photo added',
  INCIDENT_RESOLVED: 'Resolved',
  INCIDENT_CLOSED: 'Closed',
  INCIDENT_REOPENED: 'Reopened',
  ADMIN_OVERRIDE: 'Admin override',
};

/**
 * Which visual family an event belongs to. Drives the marker colour only —
 * every event carries its own full text regardless.
 */
export const HISTORY_ACTION_TONE: Record<HistoryActionType, string> = {
  INCIDENT_CREATED: 'start',
  INCIDENT_TRIAGED: 'status',
  PRIORITY_CHANGED: 'priority',
  SLA_CREATED: 'sla',
  SLA_UPDATED: 'sla',
  AGENT_ASSIGNED: 'agent',
  AGENT_REASSIGNED: 'agent',
  AGENT_UNASSIGNED: 'agent',
  STATUS_CHANGED: 'status',
  AGENT_AVAILABILITY_CHANGED: 'agent',
  IMAGE_ADDED: 'image',
  INCIDENT_RESOLVED: 'done',
  INCIDENT_CLOSED: 'done',
  INCIDENT_REOPENED: 'warn',
  ADMIN_OVERRIDE: 'warn',
};

/** Glyph shown in the timeline marker. Decorative — always paired with text. */
export const HISTORY_ACTION_ICONS: Record<HistoryActionType, string> = {
  INCIDENT_CREATED: '\u25CF',
  INCIDENT_TRIAGED: '\u25C6',
  PRIORITY_CHANGED: '\u2191',
  SLA_CREATED: '\u23F1',
  SLA_UPDATED: '\u23F1',
  AGENT_ASSIGNED: '\u25B8',
  AGENT_REASSIGNED: '\u21C4',
  AGENT_UNASSIGNED: '\u2205',
  STATUS_CHANGED: '\u2192',
  AGENT_AVAILABILITY_CHANGED: '\u25CB',
  IMAGE_ADDED: '\u25A3',
  INCIDENT_RESOLVED: '\u2713',
  INCIDENT_CLOSED: '\u2714',
  INCIDENT_REOPENED: '\u21BA',
  ADMIN_OVERRIDE: '\u26A0',
};

/** One entry in an incident's audit trail. Read-only: the API has no writer. */
export interface HistoryEvent {
  id: number;
  incident_id: number;
  actor_name?: string | null;
  actor_id?: number | null;
  actor_role: ActorRole;
  action: HistoryActionType;
  old_value?: string | null;
  new_value?: string | null;
  description?: string | null;
  created_at: string;
}

export interface UserInfo {
  user_id: number;
  name: string;
  role: Role;
  access_token: string;
}

export interface Incident {
  id: number;
  title: string;
  description?: string | null;
  category: string;
  source: string;
  status: IncidentStatus;

  /** System-assigned. Citizens never choose this. */
  priority_level: PriorityLevel;
  priority_label: string;
  severity: string;
  priority: number;

  /** SLA — visible to every role, editable only by an admin. */
  sla_hours?: number | null;
  sla_deadline?: string | null;
  sla_status?: SlaStatus | null;

  latitude: number;
  longitude: number;
  reported_by?: number | null;
  reported_by_name?: string | null;
  assigned_agent_id?: number | null;
  assigned_agent_name?: string | null;
  image_count: number;
  created_at: string;
  updated_at: string;
}

/**
 * What an attached image is evidence of. Mirrors ImageKind on the backend.
 *
 * REPORT     — the citizen's photo of the problem.
 * RESOLUTION — the assigned agent's photo of the finished work.
 *
 * Both are stored and served identically; this only decides the label.
 */
export type ImageKind = 'REPORT' | 'RESOLUTION';

export const IMAGE_KIND_LABELS: Record<ImageKind, string> = {
  REPORT: 'Reported problem',
  RESOLUTION: 'Proof of work',
};

export interface ImageMeta {
  id: number;
  incident_id: number;
  filename: string;
  content_type: string;
  kind: ImageKind;
  created_at: string;
}

export interface Agent {
  id: number;
  name: string;
  email: string;
  role: Role;
  is_available: boolean;
  last_assigned_at?: string | null;
  active_incident_count: number;
  created_at: string;
}

/**
 * How an account can sign in. Derived server-side from which credentials the
 * account actually holds — the credentials themselves are never sent.
 */
export type AuthMethod = 'EMAIL' | 'GOOGLE';

export const AUTH_METHOD_LABELS: Record<AuthMethod, string> = {
  EMAIL: 'Email',
  GOOGLE: 'Google',
};

export const ROLE_LABELS: Record<Role, string> = {
  USER: 'Citizen',
  AGENT: 'Agent',
  ADMIN: 'Admin',
};

/** The order roles are offered in, least privileged first. */
export const ROLES: Role[] = ['USER', 'AGENT', 'ADMIN'];

/**
 * One row of the admin user-management table.
 *
 * Deliberately carries no credential material: no password hash, no Google
 * subject id, no tokens. The backend does not send them.
 */
export interface AdminUser {
  id: number;
  name: string;
  email: string;
  role: Role;
  is_active: boolean;
  auth_methods: AuthMethod[];
  created_at: string;
  last_login_at?: string | null;
  is_available: boolean;
}

/** Mirrors UserAuditAction on the backend. */
export type UserAuditActionType =
  | 'USER_REGISTERED'
  | 'USER_ROLE_CHANGED'
  | 'USER_STATUS_CHANGED'
  | 'GOOGLE_ACCOUNT_LINKED';

export const USER_AUDIT_LABELS: Record<UserAuditActionType, string> = {
  USER_REGISTERED: 'Account created',
  USER_ROLE_CHANGED: 'Role changed',
  USER_STATUS_CHANGED: 'Status changed',
  GOOGLE_ACCOUNT_LINKED: 'Google account linked',
};

/**
 * One entry of the account audit trail.
 *
 * Separate from HistoryEvent on purpose: that answers "what happened to
 * incident N", this answers "what happened to this account". Read-only — the
 * API has no writer, and the table rejects UPDATE and DELETE.
 */
export interface UserAuditEvent {
  id: number;
  action: UserAuditActionType;
  actor_name?: string | null;
  actor_email?: string | null;
  actor_role: ActorRole;
  target_user_id?: number | null;
  target_name?: string | null;
  target_email?: string | null;
  old_value?: string | null;
  new_value?: string | null;
  description?: string | null;
  created_at: string;
}

/** Thrown by apiFetch so callers can branch on the HTTP status. */
export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

// ── Core fetch wrapper ──────────────────────────────────────
async function readError(res: Response): Promise<string> {
  const body = await res.json().catch(() => null);
  if (!body) return res.statusText || `Request failed (${res.status})`;
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d: { msg?: string }) => d?.msg ?? JSON.stringify(d))
      .join('; ');
  }
  return detail ? JSON.stringify(detail) : res.statusText;
}

function onUnauthenticated(): never {
  clearToken();
  if (typeof window !== 'undefined') window.location.href = '/login';
  throw new ApiError('Your session has expired. Please sign in again.', 401);
}

async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    ...((options.headers as Record<string, string>) || {}),
  };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  if (options.body && typeof options.body === 'string') {
    headers['Content-Type'] = 'application/json';
  }

  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, { ...options, headers });
  } catch {
    throw new ApiError(
      `Cannot reach the UrbanEye+ API at ${BASE_URL}. Is the backend running?`,
      0,
    );
  }

  if (res.status === 401) onUnauthenticated();
  if (!res.ok) throw new ApiError(await readError(res), res.status);
  if (res.status === 204) return undefined as T;
  return res.json();
}

// ── Auth ─────────────────────────────────────────────────────
export async function apiLogin(email: string, password: string): Promise<UserInfo> {
  const data = await apiFetch<{
    access_token: string;
    token_type: string;
    role: Role;
    user_id: number;
    name: string;
  }>('/api/v1/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
  const userInfo: UserInfo = {
    user_id: data.user_id,
    name: data.name,
    role: data.role,
    access_token: data.access_token,
  };
  setToken(data.access_token);
  setStoredUser(userInfo);
  return userInfo;
}

/**
 * Create an account and sign in with it.
 *
 * There is no role argument, and adding one to the body would achieve nothing:
 * the backend hard-codes USER for every registration and its request schema has
 * no role field. Promotion to AGENT or ADMIN happens only in Admin → User
 * Management, by an authenticated admin.
 */
export async function apiRegister(
  name: string,
  email: string,
  password: string,
): Promise<UserInfo> {
  const data = await apiFetch<{
    access_token: string;
    role: Role;
    user_id: number;
    name: string;
  }>('/api/v1/auth/register', {
    method: 'POST',
    body: JSON.stringify({ name, email, password }),
  });
  const userInfo: UserInfo = {
    user_id: data.user_id,
    name: data.name,
    role: data.role,
    access_token: data.access_token,
  };
  setToken(data.access_token);
  setStoredUser(userInfo);
  return userInfo;
}

export async function apiMe(): Promise<AdminUser> {
  return apiFetch<AdminUser>('/api/v1/auth/me');
}

/** Which sign-in methods this server is configured for. */
export async function apiAuthConfig(): Promise<{ google_enabled: boolean }> {
  return apiFetch<{ google_enabled: boolean }>('/api/v1/auth/config');
}

/**
 * Start Google sign-in.
 *
 * A full-page navigation, not a fetch: the backend needs to set the HttpOnly
 * state cookie that its callback checks, and Google's consent screen cannot be
 * loaded in an XHR. The browser comes back to /auth/callback with a token.
 */
export function startGoogleSignIn(next = '/auth/callback') {
  window.location.href = `${BASE_URL}/api/v1/auth/google/login?next=${encodeURIComponent(next)}`;
}

/**
 * Adopt the session handed back by the Google callback.
 *
 * The role in the URL is only used to pick a landing page; every protected
 * request is authorized by the backend from the token, so a tampered role
 * parameter buys nothing beyond a redirect to a page that will refuse to load.
 */
export function adoptSession(info: UserInfo) {
  setToken(info.access_token);
  setStoredUser(info);
}

// ── Incidents ─────────────────────────────────────────────────
export async function apiGetIncidents(
  params?: Record<string, string>,
): Promise<Incident[]> {
  const qs = params ? '?' + new URLSearchParams(params).toString() : '';
  return apiFetch<Incident[]>(`/api/v1/incidents/${qs}`);
}

export async function apiGetIncident(id: number): Promise<Incident> {
  return apiFetch<Incident>(`/api/v1/incidents/${id}`);
}

/**
 * Create an incident.
 *
 * Note there is no severity or priority argument: priority, severity and the
 * SLA are decided server-side from the incident's category. Anything the
 * client sends for those fields is stripped by the API.
 */
export async function apiCreateIncident(
  data: {
    title: string;
    description?: string;
    category: string;
    latitude: number;
    longitude: number;
  },
  images: File[],
): Promise<Incident> {
  const token = getToken();
  const formData = new FormData();
  formData.append('data', JSON.stringify(data));
  for (const img of images) formData.append('images', img);

  let res: Response;
  try {
    res = await fetch(`${BASE_URL}/api/v1/incidents/`, {
      method: 'POST',
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: formData,
    });
  } catch {
    throw new ApiError(
      `Cannot reach the UrbanEye+ API at ${BASE_URL}. Is the backend running?`,
      0,
    );
  }
  if (res.status === 401) onUnauthenticated();
  if (!res.ok) throw new ApiError(await readError(res), res.status);
  return res.json();
}

export async function apiUpdateStatus(
  id: number,
  status: IncidentStatus,
): Promise<Incident> {
  return apiFetch<Incident>(`/api/v1/incidents/${id}/status`, {
    method: 'PATCH',
    body: JSON.stringify({ status }),
  });
}

/**
 * Assign or unassign an agent (admin only).
 *
 * The backend refuses an unavailable agent with HTTP 422 unless
 * `overrideAvailability` is true — the override is validated server-side, so
 * disabling the option in the UI is a convenience, not the control.
 */
export async function apiAssignAgent(
  incidentId: number,
  agentId: number | null,
  overrideAvailability = false,
): Promise<Incident> {
  return apiFetch<Incident>(`/api/v1/incidents/${incidentId}/assign`, {
    method: 'PUT',
    body: JSON.stringify({
      agent_id: agentId,
      override_availability: overrideAvailability,
    }),
  });
}

/** Admin only: override the SLA window. Pass null to clear it. */
export async function apiUpdateSla(
  incidentId: number,
  slaHours: number | null,
): Promise<Incident> {
  return apiFetch<Incident>(`/api/v1/incidents/${incidentId}/sla`, {
    method: 'PATCH',
    body: JSON.stringify({ sla_hours: slaHours }),
  });
}

/** Admin only: override the system-assigned priority. */
export async function apiUpdatePriority(
  incidentId: number,
  priorityLevel: PriorityLevel,
): Promise<Incident> {
  return apiFetch<Incident>(`/api/v1/incidents/${incidentId}/priority`, {
    method: 'PATCH',
    body: JSON.stringify({ priority_level: priorityLevel }),
  });
}

// ── History / audit trail ─────────────────────────────────────
/**
 * An incident's audit trail, oldest event first.
 *
 * Authorization matches reading the incident itself: a citizen gets their own
 * incidents, an agent gets the ones assigned to them, an admin gets any. There
 * is deliberately no writer here — history is produced by the backend as a
 * side effect of real operations and cannot be posted, edited or deleted.
 */
export async function apiGetHistory(incidentId: number): Promise<HistoryEvent[]> {
  return apiFetch<HistoryEvent[]>(`/api/v1/incidents/${incidentId}/history`);
}

// ── Images ────────────────────────────────────────────────────
export async function apiGetImages(incidentId: number): Promise<ImageMeta[]> {
  return apiFetch<ImageMeta[]>(`/api/v1/incidents/${incidentId}/images`);
}

/**
 * Attach proof that the work was completed.
 *
 * Agent-only in practice: the backend accepts this from the assigned agent (or
 * an admin) and only once the incident is RESOLVED — a citizen gets 403 and an
 * early upload gets 422, whatever the UI offers. The image lands in the same
 * store as a report photo and becomes visible to the reporter, the assigned
 * agent, and admins.
 */
export async function apiUploadResolutionImage(
  incidentId: number,
  file: File,
): Promise<ImageMeta> {
  const token = getToken();
  const formData = new FormData();
  formData.append('file', file);

  let res: Response;
  try {
    res = await fetch(
      `${BASE_URL}/api/v1/incidents/${incidentId}/images/resolution`,
      {
        method: 'POST',
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        body: formData,
      },
    );
  } catch {
    throw new ApiError(
      `Cannot reach the UrbanEye+ API at ${BASE_URL}. Is the backend running?`,
      0,
    );
  }
  if (res.status === 401) onUnauthenticated();
  if (!res.ok) throw new ApiError(await readError(res), res.status);
  return res.json();
}

/**
 * Fetch image bytes with the bearer token and return a blob object URL.
 *
 * `<img src>` cannot carry an Authorization header, and the backend has no
 * query-string token, so a plain URL always 401s. Callers MUST pass the
 * returned URL to URL.revokeObjectURL when the image is no longer displayed.
 */
export async function fetchImageObjectUrl(
  incidentId: number,
  imageId: number,
  signal?: AbortSignal,
): Promise<string> {
  const token = getToken();
  const res = await fetch(
    `${BASE_URL}/api/v1/incidents/${incidentId}/images/${imageId}`,
    {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      signal,
    },
  );
  if (res.status === 401) onUnauthenticated();
  if (!res.ok) throw new ApiError(await readError(res), res.status);
  return URL.createObjectURL(await res.blob());
}

// ── Agents ────────────────────────────────────────────────────
export async function apiGetAgents(): Promise<Agent[]> {
  return apiFetch<Agent[]>('/api/v1/agents/');
}

/** The signed-in agent's own profile, including persisted availability. */
export async function apiGetMyAgentProfile(): Promise<Agent> {
  return apiFetch<Agent>('/api/v1/agents/me');
}

/** An agent setting their own availability. Persists across sessions. */
export async function apiSetAvailability(isAvailable: boolean): Promise<Agent> {
  return apiFetch<Agent>('/api/v1/agents/availability', {
    method: 'POST',
    body: JSON.stringify({ is_available: isAvailable }),
  });
}

/** Admin overriding another agent's availability. */
export async function apiSetAgentAvailability(
  agentId: number,
  isAvailable: boolean,
): Promise<Agent> {
  return apiFetch<Agent>(`/api/v1/agents/${agentId}/availability`, {
    method: 'PATCH',
    body: JSON.stringify({ is_available: isAvailable }),
  });
}

// ── Admin: user management ────────────────────────────────────
/**
 * Every account, newest first. ADMIN only.
 *
 * A USER or AGENT calling this receives 403 and an unauthenticated caller 401 —
 * hiding the navigation entry is a courtesy, not the control.
 */
export async function apiGetUsers(params?: {
  role?: Role;
  q?: string;
}): Promise<AdminUser[]> {
  const search = new URLSearchParams();
  if (params?.role) search.set('role', params.role);
  if (params?.q) search.set('q', params.q);
  const qs = search.toString();
  return apiFetch<AdminUser[]>(`/api/v1/admin/users${qs ? `?${qs}` : ''}`);
}

/** Grant a role to another user. ADMIN only; refused (409) for your own account. */
export async function apiSetUserRole(userId: number, role: Role): Promise<AdminUser> {
  return apiFetch<AdminUser>(`/api/v1/admin/users/${userId}/role`, {
    method: 'PATCH',
    body: JSON.stringify({ role }),
  });
}

/** Activate or deactivate an account. ADMIN only. */
export async function apiSetUserActive(
  userId: number,
  isActive: boolean,
): Promise<AdminUser> {
  return apiFetch<AdminUser>(`/api/v1/admin/users/${userId}/status`, {
    method: 'PATCH',
    body: JSON.stringify({ is_active: isActive }),
  });
}

/** Recent account/authorization events, newest first. ADMIN only. */
export async function apiGetUserAudit(userId?: number): Promise<UserAuditEvent[]> {
  const qs = userId ? `?user_id=${userId}` : '';
  return apiFetch<UserAuditEvent[]>(`/api/v1/admin/audit${qs}`);
}

// ── Formatting helpers ────────────────────────────────────────
export function formatDateTime(iso?: string | null): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function formatDate(iso?: string | null): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleDateString(undefined, {
    month: 'short',
    day: '2-digit',
    year: 'numeric',
  });
}

/** "18:42" — the time of day, for dense timeline rows. */
export function formatTime(iso?: string | null): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '—';
  return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

/** "in 3h 20m" / "4h 10m overdue" — relative to an SLA deadline. */
export function formatDeadlineDelta(iso?: string | null): string {
  if (!iso) return '';
  const deadline = new Date(iso).getTime();
  if (Number.isNaN(deadline)) return '';
  const diffMinutes = Math.round((deadline - Date.now()) / 60000);
  const overdue = diffMinutes < 0;
  const total = Math.abs(diffMinutes);
  const days = Math.floor(total / 1440);
  const hours = Math.floor((total % 1440) / 60);
  const minutes = total % 60;

  const parts: string[] = [];
  if (days) parts.push(`${days}d`);
  if (hours) parts.push(`${hours}h`);
  if (!days && minutes) parts.push(`${minutes}m`);
  const span = parts.length ? parts.join(' ') : 'under a minute';

  return overdue ? `${span} overdue` : `${span} remaining`;
}
