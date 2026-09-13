# UrbanEye+ — Authentication & Role Management

How people get into UrbanEye+, and how they get the role they hold.

---

## 1. The rule everything else follows

**A client never chooses its own role.**

`POST /api/v1/auth/register` creates a `USER`. So does a first Google sign-in.
A body like

```json
{ "email": "...", "password": "...", "role": "ADMIN" }
```

produces a `USER` account: the `UserCreate` schema has no `role` field, pydantic
drops the unknown key, and `register()` passes `UserRole.USER` as a literal.
The same holds for a `?role=` query parameter and for anything in
`localStorage`.

Roles change in exactly one place — `user_service.set_user_role` — reachable
only through `PATCH /api/v1/admin/users/{id}/role`, behind `require_admin`.

**Authorization is read from the database, never from the token.**
`get_current_user` decodes the JWT only to learn *who* is calling; it then loads
that user's row, and `require_admin` / `require_agent` check the role on that
row. The `role` claim inside the JWT is a convenience for the client. Two
consequences worth knowing:

- A promotion takes effect on the user's **very next request** — they do not
  need to sign out and back in.
- A demotion or deactivation revokes access **immediately**, without waiting for
  the token to expire.

---

## 2. Sign-in methods

### Email + password

| | |
|---|---|
| Hashing | bcrypt, via `passlib` (unchanged from before this feature) |
| Stored | the digest only — the plain password is never written or logged |
| Password rules | 8–128 characters, at least one letter and one number |
| Email | validated by `EmailStr`, stored and matched case-insensitively |
| Duplicates | `409 Conflict`, case-insensitive (`Riya@x.com` = `riya@x.com`) |
| Wrong password / unknown account | both `401` with the **same** message, so responses cannot be used to enumerate registered addresses |
| Deactivated account | `403` with a clear message — only ever shown to someone who already proved they hold the password |

Registration returns a session token, so a new user lands on their dashboard
rather than being sent back to retype what they just typed.

### Google Sign-In

The OAuth 2.0 **authorization code** flow, run server-side:

```
browser → GET /api/v1/auth/google/login
            ├─ mints a signed, 10-minute `state` token
            ├─ sets it as an HttpOnly, SameSite=Lax cookie
            └─ 307 → accounts.google.com/o/oauth2/v2/auth

           (user signs in with Google)

Google  → GET /api/v1/auth/google/callback?code=…&state=…
            ├─ state must match the cookie   ← CSRF control
            ├─ state must be ours, unexpired
            ├─ code exchanged for tokens at oauth2.googleapis.com (server-side,
            │  with the client secret — which never reaches the browser)
            ├─ the ID token's signature, issuer, audience and expiry are
            │  verified by google-auth against Google's certificates
            ├─ email_verified must be true
            └─ 307 → FRONTEND_URL/auth/callback?token=…
```

Google's access and refresh tokens are used for nothing beyond the exchange —
we request identity scopes only — so they are never stored.

**The email's domain never affects the role.** `@gmail.com`, `@vit.ac.in`,
`@company.com` — all become `USER`.

---

## 3. Account linking

One person, one account. Resolution order on every Google sign-in:

1. **By Google `sub`.** The `sub` is stable for the life of the Google account
   and survives the user changing their Google email address — so an email
   change signs the same person into the same account instead of forking one.
2. **By verified email.** Someone who registered with a password and later
   clicks *Continue with Google* is **linked**, not duplicated: the Google
   identity is attached to their existing account, which keeps its role, its
   password, its id, and its incidents. Only a *verified* Google email can
   reach this branch.
3. **Otherwise**, a new `USER` account.

Two unrelated accounts can never be merged: `google_subject_id` is `UNIQUE`, so
a `sub` binds to exactly one row, and step 2 only ever attaches a `sub` to a row
that has none.

After linking, **both** sign-in methods work. The admin table shows
`Email + Google`.

An account created purely by Google has `password_hash = NULL`, and
`authenticate_user` refuses it outright rather than comparing against an empty
string — no password signs it in.

---

## 4. Roles

| Role | Can |
|---|---|
| `USER` | report incidents, see their own |
| `AGENT` | work assigned incidents, set their availability |
| `ADMIN` | everything, plus **User Management** |

New accounts → `USER`, always. Promotion is `Admin → User Management`.

### Lockout protection

The system must always retain at least one administrator who can sign in. Two
rules, enforced in the service layer (so they hold for any caller) and surfaced
as HTTP `409`:

1. **An admin cannot act on their own account** — cannot change their own role,
   cannot deactivate themselves. There is no legitimate case for it; another
   admin can always perform the change.
2. **The last *active* admin cannot be demoted or deactivated.** A deactivated
   admin does not count as one, because they cannot sign in.

Demoting *other* admins is allowed — an organisation must be able to revoke
admin access — and the two rules above guarantee a survivor.

---

## 5. Audit trail

Role changes go to **`user_audit_log`**, not to `incident_history`.

`incident_history` answers "what happened to incident N", and every row there is
keyed to an incident. An account being promoted belongs to no incident, so
putting it there would need a fake incident id. Two questions, two tables.

Recorded: `USER_REGISTERED`, `USER_ROLE_CHANGED`, `USER_STATUS_CHANGED`,
`GOOGLE_ACCOUNT_LINKED` — with actor, target, old/new value and a sentence.

**Not** recorded: passwords, hashes, Google tokens, authorization codes, or the
`sub` claim. The trail records that an identity was linked, not the credential
that proved it.

Append-only, three overlapping ways: no write endpoint (GET only), SQLAlchemy
mapper events that raise on UPDATE/DELETE, and a database trigger.

Read it at `GET /api/v1/admin/audit`, or in the *Account activity* section of
User Management.

---

## 6. Setting up Google credentials

Google sign-in is **optional**. Leave the two variables blank and the app runs
normally with email/password — the Google button is simply not shown, and the
endpoints answer `503`.

1. Go to <https://console.cloud.google.com/apis/credentials>.
2. Create (or pick) a project.
3. **OAuth consent screen** → External → fill in app name and support email.
   While the app is in *Testing*, add your own Google address under
   **Test users**, or sign-in will be refused.
4. **Credentials → Create credentials → OAuth client ID**
   - Application type: **Web application**
   - **Authorised redirect URI** — exactly, including the port and path:
     ```
     http://localhost:8000/api/v1/auth/google/callback
     ```
     A mismatch here is the most common failure; Google reports it as
     `redirect_uri_mismatch`, which the login page shows verbatim.
   - Authorised JavaScript origins are **not** needed — the browser never talks
     to Google's token endpoint; the backend does.
5. Copy the client ID and secret into `.env`:

```bash
GOOGLE_CLIENT_ID=<your-id>.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=<your-secret>
GOOGLE_REDIRECT_URI=http://localhost:8000/api/v1/auth/google/callback
FRONTEND_URL=http://localhost:3000
```

Restart the backend. `GET /api/v1/auth/config` should now report
`{"google_enabled": true}` and the button appears on the login page.

Never commit real credentials. `.env` is gitignored; `.env.example` holds
placeholders only.

---

## 7. Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_CLIENT_ID` | *(blank)* | OAuth client ID. Blank disables Google sign-in. |
| `GOOGLE_CLIENT_SECRET` | *(blank)* | OAuth client secret. **Server-side only** — never sent to the browser. |
| `GOOGLE_REDIRECT_URI` | `http://localhost:8000/api/v1/auth/google/callback` | Must match the console entry exactly. |
| `FRONTEND_URL` | `http://localhost:3000` | Where the callback returns the browser, and the only origin a post-login redirect may target. |
| `OAUTH_STATE_EXPIRE_SECONDS` | `600` | Lifetime of the CSRF `state` token. |
| `COOKIE_SECURE` | `false` | Set `true` behind HTTPS. `false` locally, or `http://localhost` drops the cookie. |
| `SECRET_KEY` | — | Signs both JWTs and OAuth state. Change it in any real deployment. |

---

## 8. Bootstrapping the first admin

Ordinary users no longer need a script — they register in the app. But the
*first* admin cannot promote themselves into existence, so one bootstrap path
remains:

```bash
cd backend
source .venv/bin/activate
python scripts/create_demo_data.py
```

This seeds `admin@urbaneye.local` / `Admin@1234` (plus demo agent and citizen
accounts) and is safe to re-run — existing accounts are skipped, never
overwritten. Change the credentials via `DEMO_ADMIN_*` in `.env` before using
this anywhere but a development machine.

From then on: sign in as that admin, open **User Management**, and promote real
people. The script is needed only once.

To promote an account directly in the database instead:

```sql
UPDATE users SET role = 'ADMIN' WHERE email = 'you@example.com';
```

---

## 9. Security properties

| Control | Status | How |
|---|---|---|
| Password hashing | bcrypt via passlib | `core/security.py`; verified by a test that reads the stored row |
| Plaintext passwords | never stored or echoed | tested |
| Client-chosen role | impossible | no `role` field in the schema; tested against body **and** query string |
| Admin API authorization | server-side on every endpoint | `require_admin`; tested for 401 / 403 / 200 |
| Privilege escalation by a user | blocked | tested, including PATCHing one's own row |
| Role change latency | immediate | role read from the DB row, not the JWT claim; tested with a stale token |
| Self-demotion lockout | blocked | `409`; tested |
| Last-admin lockout | blocked | `409`, counting only *active* admins; tested |
| OAuth CSRF | signed `state` + HttpOnly SameSite=Lax cookie, compared with `compare_digest` | tested: missing cookie and mismatched state both refused |
| Redirect-URI validation | Google enforces the registered URI; `next` is reduced to a same-site path | tested against `https://`, `//host`, `http://` |
| ID token verification | signature, issuer, audience, expiry via `google-auth` | `services/google_oauth.py` |
| Unverified Google email | refused | tested |
| Account enumeration | one message for wrong password and unknown address | tested |
| Duplicate accounts | `409`, case-insensitive; Google linking rather than forking | tested |
| Inactive accounts | cannot log in; outstanding tokens rejected on the next request | tested |
| Token expiry | `JWT_EXPIRE_MINUTES` (default 24h) | existing behaviour |
| Secrets in responses | none — no hash, no `sub`, no tokens in any payload | tested on the admin list |
| Cookie flags | HttpOnly, SameSite=Lax, `Secure` via `COOKIE_SECURE` | verified |

### Known limitations

- **A real Google sign-in has not been performed.** Everything on our side of
  the boundary is tested — state, verification logic, account resolution,
  linking, the constructed authorization URL, and the refusal of a forged code
  (Google's token endpoint rejected it with `invalid_client`). Completing an
  actual consent screen needs real credentials and an interactive login. See
  §10 of the README's manual test list.
- **JWTs are stateless**, so "sign out everywhere" is not instant for a token
  that is merely old. Deactivating the account *is* instant, because
  `is_active` is re-read on every request.
- **No password reset and no email verification.** Deliberately out of scope;
  an admin can deactivate an account, and a user's address is either
  self-asserted (password sign-up) or verified by Google.
- `COOKIE_SECURE=false` is correct for local HTTP only. Set it `true` before
  serving over HTTPS.
