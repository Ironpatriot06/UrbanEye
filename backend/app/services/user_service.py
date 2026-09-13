"""
User service layer.

All database operations for users and authentication live here.

Authorization invariant
-----------------------
`create_user` takes the role as an argument that only server code supplies, and
every caller in the application passes UserRole.USER.  There is no path from a
request body to this argument — see app/api/auth.py, where the registration
payload schema has no role field at all.  Roles change in exactly one place,
`set_user_role`, which is reachable only through an ADMIN-guarded endpoint.

Lockout invariant
-----------------
The system must always retain at least one usable administrator.  `set_user_role`
and `set_user_active` both refuse any change that would leave zero active
ADMINs, and both refuse to let an administrator act on their own account.
"""

from datetime import datetime, timezone
from typing import List, Optional, Tuple

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.core.security import hash_password, verify_password
from app.models.user import User, UserRole
from app.models.user_audit import UserAuditAction
from app.schemas.user import UserCreate
from app.services import user_audit_service


class RoleChangeError(RuntimeError):
    """
    Raised when a role or status change is refused by a safety rule.

    Carries the message shown to the admin, so the router can turn it straight
    into an HTTP 409 without re-deriving why the change was rejected.
    """


def normalize_email(email: str) -> str:
    """
    Canonical form of an email address for storage and lookup.

    Case-folding the whole address is a small over-reach — the local part is
    technically case-sensitive — but it is what users expect, and it is what
    stops Riya@x.com and riya@x.com becoming two accounts with two different
    roles.
    """
    return email.strip().lower()


def create_user(
    db: Session,
    payload: UserCreate,
    role: UserRole = UserRole.USER,
    *,
    actor: Optional[User] = None,
) -> User:
    """
    Create and persist a password-authenticated user.

    `role` is a server-supplied argument, never a client-supplied one: the
    UserCreate schema has no role field, so a request body cannot reach it.
    The password is bcrypt-hashed here and the plain text is not retained.
    """
    user = User(
        name=payload.name,
        email=normalize_email(payload.email),
        password_hash=hash_password(payload.password),
        role=role,
        is_active=True,
        is_available=True,
    )
    db.add(user)
    db.flush()

    user_audit_service.record(
        db,
        UserAuditAction.USER_REGISTERED,
        target=user,
        actor=actor,
        new_value=role.value,
        description=(
            f"Account created for {user.email} via email and password "
            f"with the {role.value} role."
        ),
    )

    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """
    Return the User with the given email, or None.

    Matched case-insensitively so that signing in as Riya@x.com finds the
    account stored as riya@x.com, and so a Google identity whose email differs
    only in case links to the existing account instead of forking a new one.
    """
    if not email:
        return None
    return (
        db.query(User)
        .filter(func.lower(User.email) == normalize_email(email))
        .first()
    )


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Return the User with the given id, or None."""
    return db.query(User).filter(User.id == user_id).first()


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """
    Return the User if credentials are valid, else None.

    Timing-safe: we always run the hash comparison even if the email is not
    found, to avoid email-enumeration via response timing.
    """
    user = get_user_by_email(db, email)
    if user is None:
        return None
    if not user.password_hash:
        # A Google-only account.  There is no stored digest to compare, and
        # treating the absent hash as an empty password would let anyone in.
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def get_all_agents(db: Session) -> list[User]:
    """Return all users with the AGENT role, with their current workload attached.

    `active_incident_count` is a transient attribute (not an ORM column) read by
    the UserRead schema so the admin dashboard can tell an idle available agent
    apart from a busy one.
    """
    from app.services.incident_service import count_open_incidents_for_agent

    agents = (
        db.query(User)
        .filter(User.role == UserRole.AGENT)
        .order_by(User.name)
        .all()
    )
    for agent in agents:
        agent.active_incident_count = count_open_incidents_for_agent(db, agent.id)
    return agents


def get_agent_by_id(db: Session, agent_id: int) -> Optional[User]:
    """Return the AGENT-role user with the given id, or None.

    Returns None for a non-agent user so callers cannot flip availability on an
    admin or citizen account.
    """
    return (
        db.query(User)
        .filter(User.id == agent_id, User.role == UserRole.AGENT)
        .first()
    )


def set_agent_availability(
    db: Session,
    agent: User,
    is_available: bool,
    actor: Optional[User] = None,
) -> User:
    """
    Persist an agent's availability flag.

    This is the ONLY place availability changes.  Authentication deliberately
    does not touch it: an agent who marked themselves unavailable stays
    unavailable across logout and the next login, until they or an admin change
    it explicitly.

    Audit trail
    -----------
    Availability belongs to the agent, not to any one incident, but the audit
    trail is per-incident — so the event is recorded against each incident
    still open on that agent's plate, which is exactly where it is
    operationally relevant: those are the incidents whose handling it affects.
    Finished incidents are left alone; their outcome cannot change now.

    `actor` distinguishes an agent setting their own status from an admin
    overriding it, which is the difference the admin audit view needs to show.
    A call that does not change the flag records nothing.
    """
    from app.models.history import HistoryAction
    from app.models.incident import SLA_FREEZE_STATUSES, Incident
    from app.services import history_service

    changed = agent.is_available != is_available
    agent.is_available = is_available

    if changed:
        open_incidents = (
            db.query(Incident)
            .filter(
                Incident.assigned_agent_id == agent.id,
                Incident.status.notin_(list(SLA_FREEZE_STATUSES)),
            )
            .all()
        )
        state = "Available" if is_available else "Unavailable"
        by_self = actor is not None and actor.id == agent.id
        who = "themselves" if by_self else (actor.name if actor else "the system")

        for incident in open_incidents:
            history_service.record(
                db,
                incident.id,
                HistoryAction.AGENT_AVAILABILITY_CHANGED,
                actor=actor,
                old_value="Available" if not is_available else "Unavailable",
                new_value=state,
                description=(
                    f"{agent.name}, the agent on this incident, was marked "
                    f"{state.lower()} by {who}."
                ),
            )

    db.commit()
    db.refresh(agent)
    return agent


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def touch_last_login(db: Session, user: User) -> User:
    """
    Stamp the user's last successful sign-in.

    Deliberately the only thing login writes to the user row.  In particular it
    does not touch `is_available`: an agent who marked themselves unavailable
    stays unavailable across a logout and the next login.
    """
    user.last_login_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# Google identities
# ---------------------------------------------------------------------------

def get_user_by_google_subject(db: Session, subject: str) -> Optional[User]:
    """Return the User linked to a Google `sub`, or None."""
    if not subject:
        return None
    return db.query(User).filter(User.google_subject_id == subject).first()


def resolve_google_user(db: Session, identity) -> Tuple[User, bool, bool]:
    """
    Map a verified Google identity to an application account.

    Returns (user, created, linked).

    Resolution order, and why
    -------------------------
    1. By Google `sub`.  The sub is stable for the life of the Google account
       and survives the user changing their Google email address, so matching
       on it first means an email change signs the same person into the same
       account instead of forking a new one.
    2. By email.  A user who registered with a password and later clicks
       "Continue with Google" arrives here: the verified Google email matches
       their existing account, so the Google identity is LINKED to it rather
       than creating a second account for the same person.  Only a verified
       email can reach this branch — google_oauth refuses unverified ones —
       because an unverified address proves nothing about who owns the mailbox
       and would otherwise be an account-takeover path.
    3. Otherwise a new account, always with role USER.

    Two unrelated accounts can never be merged by this function: step 1 binds a
    sub to exactly one row (the column is UNIQUE), and step 2 only ever attaches
    a sub to a row that has none.  An account that already carries a different
    sub is left alone and the sign-in is refused by the caller.

    The role is never read from, or influenced by, the Google profile — not by
    the email's domain, not by any claim.  Existing accounts keep whatever role
    they already have; new ones get USER.
    """
    # 1. Known Google identity.
    user = get_user_by_google_subject(db, identity.subject)
    if user is not None:
        # Keep the stored email in step with Google if the user changed it
        # there, unless that address already belongs to another account.
        if normalize_email(user.email) != identity.email:
            clash = get_user_by_email(db, identity.email)
            if clash is None:
                user.email = identity.email
        db.commit()
        db.refresh(user)
        return user, False, False

    # 2. Existing account with the same verified email — link, do not duplicate.
    user = get_user_by_email(db, identity.email)
    if user is not None:
        if user.google_subject_id and user.google_subject_id != identity.subject:
            # Should be unreachable (the sub lookup above would have matched),
            # but refusing beats silently rebinding somebody else's identity.
            raise RoleChangeError(
                "This email address is already linked to a different Google account."
            )
        user.google_subject_id = identity.subject
        user_audit_service.record(
            db,
            UserAuditAction.GOOGLE_ACCOUNT_LINKED,
            target=user,
            actor=None,
            new_value="GOOGLE",
            description=(
                f"Google sign-in was linked to the existing {user.role.value} "
                f"account {user.email}. The account's role was not changed."
            ),
        )
        db.commit()
        db.refresh(user)
        return user, False, True

    # 3. Brand new account.  USER, unconditionally.
    user = User(
        name=identity.name,
        email=identity.email,
        password_hash=None,
        google_subject_id=identity.subject,
        role=UserRole.USER,
        is_active=True,
        is_available=True,
    )
    db.add(user)
    db.flush()
    user_audit_service.record(
        db,
        UserAuditAction.USER_REGISTERED,
        target=user,
        actor=None,
        new_value=UserRole.USER.value,
        description=(
            f"Account created for {user.email} via Google sign-in with the "
            f"USER role."
        ),
    )
    db.commit()
    db.refresh(user)
    return user, True, False


# ---------------------------------------------------------------------------
# Admin user management
# ---------------------------------------------------------------------------

def list_users(
    db: Session,
    *,
    role: Optional[UserRole] = None,
    search: Optional[str] = None,
) -> List[User]:
    """
    Every account, newest first, optionally filtered by role or a name/email search.

    Newest first because the admin's usual reason for opening this page is "who
    just signed up and needs a role".
    """
    query = db.query(User)
    if role is not None:
        query = query.filter(User.role == role)
    if search:
        pattern = f"%{search.strip().lower()}%"
        query = query.filter(
            or_(
                func.lower(User.name).like(pattern),
                func.lower(User.email).like(pattern),
            )
        )
    return query.order_by(User.created_at.desc(), User.id.desc()).all()


def count_active_admins(db: Session, *, exclude_user_id: Optional[int] = None) -> int:
    """
    How many usable administrators exist.

    "Usable" means ADMIN *and* active — a deactivated admin cannot log in, so
    counting them would let the system be left with no one able to administer
    it.  `exclude_user_id` answers "how many would remain if this one stopped
    counting".
    """
    query = db.query(User).filter(User.role == UserRole.ADMIN, User.is_active.is_(True))
    if exclude_user_id is not None:
        query = query.filter(User.id != exclude_user_id)
    return query.count()


def set_user_role(db: Session, target: User, new_role: UserRole, *, actor: User) -> User:
    """
    Change a user's role.  The only place in the application a role changes.

    Refused when
    ------------
    - the actor is the target.  An admin editing their own row is the shortest
      path to locking themselves out, and there is no legitimate case for it:
      a second admin can always perform the change.  This covers the
      "ADMIN demotes themselves" case completely rather than only when they
      happen to be the last one.
    - the target is the last active ADMIN.  Demoting them would leave nobody
      able to reach user management, and no in-app way back.  Over HTTP this
      rule is shadowed by the one above — the acting admin is themselves an
      active admin, so the target can only be the last one if it IS them — but
      it still has to hold for a caller that is not a request: a seeding
      script, a management command, or a future endpoint.

    Demotion of other admins (ADMIN -> USER / AGENT) is allowed, because the
    two rules above already guarantee a surviving administrator, and an
    organisation must be able to revoke an admin who should no longer have it.

    A no-op change is accepted and records nothing, so a double-click on the
    UI does not litter the audit trail.
    """
    if actor.id == target.id:
        raise RoleChangeError(
            "You cannot change your own role. Ask another administrator to do it."
        )

    current_role = target.role if isinstance(target.role, UserRole) else UserRole(target.role)
    if current_role == new_role:
        return target

    if current_role == UserRole.ADMIN and count_active_admins(
        db, exclude_user_id=target.id
    ) == 0:
        raise RoleChangeError(
            "This is the last active administrator. Promote another user to "
            "ADMIN before changing this account's role."
        )

    target.role = new_role
    user_audit_service.record(
        db,
        UserAuditAction.USER_ROLE_CHANGED,
        target=target,
        actor=actor,
        old_value=current_role.value,
        new_value=new_role.value,
        description=(
            f"{actor.name} changed {target.email}'s role from "
            f"{current_role.value} to {new_role.value}."
        ),
    )
    db.commit()
    db.refresh(target)
    return target


def set_user_active(db: Session, target: User, is_active: bool, *, actor: User) -> User:
    """
    Activate or deactivate an account.

    Deactivation is the reversible alternative to deletion: the account keeps
    its incidents and its place in the audit trail, but cannot sign in, and
    `get_current_user` rejects any token it already holds on the next request.

    Refused for the actor's own account and for the last active ADMIN, for the
    same lockout reasons as `set_user_role`.
    """
    if actor.id == target.id:
        raise RoleChangeError("You cannot deactivate your own account.")

    if target.is_active == is_active:
        return target

    target_role = target.role if isinstance(target.role, UserRole) else UserRole(target.role)
    if (
        not is_active
        and target_role == UserRole.ADMIN
        and count_active_admins(db, exclude_user_id=target.id) == 0
    ):
        raise RoleChangeError(
            "This is the last active administrator and cannot be deactivated."
        )

    target.is_active = is_active
    state = "active" if is_active else "deactivated"
    user_audit_service.record(
        db,
        UserAuditAction.USER_STATUS_CHANGED,
        target=target,
        actor=actor,
        old_value="ACTIVE" if not is_active else "INACTIVE",
        new_value="ACTIVE" if is_active else "INACTIVE",
        description=f"{actor.name} marked {target.email} as {state}.",
    )
    db.commit()
    db.refresh(target)
    return target
