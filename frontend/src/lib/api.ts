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

export interface ImageMeta {
  id: number;
  incident_id: number;
  filename: string;
  content_type: string;
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

export async function apiRegister(name: string, email: string, password: string) {
  return apiFetch('/api/v1/auth/register', {
    method: 'POST',
    body: JSON.stringify({ name, email, password }),
  });
}

export async function apiMe() {
  return apiFetch('/api/v1/auth/me');
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

// ── Images ────────────────────────────────────────────────────
export async function apiGetImages(incidentId: number): Promise<ImageMeta[]> {
  return apiFetch<ImageMeta[]>(`/api/v1/incidents/${incidentId}/images`);
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
