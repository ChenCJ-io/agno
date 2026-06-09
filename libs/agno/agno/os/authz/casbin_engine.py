"""Casbin adapter for the :class:`~agno.os.authz.engine.PolicyEngine` port.

This is the default (and currently only) managed-roles backend, and the ONLY
module that touches Casbin. Everything Casbin-specific — the model, the SQLAlchemy
adapter, the ``scope -> (obj, act)`` mapping, enforcement, implicit-permission
parsing — lives here, behind the agno-terms port. Swapping engines = adding a
sibling adapter, not surgery across the codebase.

Requires the optional extra: ``pip install "agno[roles]"``.
"""

from typing import Any, List, Optional, Set, Tuple

from agno.os.authz._db import engine_from_db as _engine_from_db
from agno.os.authz.engine import PolicyEngine, ScopeEntry

# Policies carry an effect (``eft``) so a role can explicitly DENY a scope. Deny
# overrides allow (``some(allow) && !some(deny)``), matching the cloud RBAC where a
# scope's value is allow|deny.
_MODEL_TEXT = """
[request_definition]
r = sub, obj, act
[policy_definition]
p = sub, obj, act, eft
[role_definition]
g = _, _
[policy_effect]
e = some(where (p.eft == allow)) && !some(where (p.eft == deny))
[matchers]
m = g(r.sub, p.sub) && (p.obj == "*" || keyMatch2(r.obj, p.obj)) && (p.act == "*" || r.act == p.act)
"""


def scope_to_obj_act(scope: str) -> Tuple[str, str]:
    """Map an agno scope string to the Casbin ``(obj, act)`` convention.

    Single source of truth for how a scope is *written* as policy and *checked*
    against a request, so both use the same spelling.

    - ``agent_os:admin``             -> ("*", "*")
    - ``sessions:write``             -> ("sessions/*", "write")   (collection/global)
    - ``agents:research-agent:run``  -> ("agents/research-agent", "run")
    - ``agents:*:run``               -> ("agents/*", "run")
    """
    if scope == "agent_os:admin":
        return ("*", "*")
    parts = scope.split(":")
    if any(part == "" for part in parts):
        raise ValueError(f"Unrecognised scope (empty component): {scope!r}")
    if len(parts) == 2:
        obj, act = f"{parts[0]}/*", parts[1]
    elif len(parts) == 3:
        obj, act = f"{parts[0]}/{parts[1]}", parts[2]
    else:
        raise ValueError(f"Unrecognised scope: {scope!r}")
    # Reject an action wildcard. In Casbin's matcher ``*`` means "all actions", but
    # the scope provider compares actions literally, so the SAME scope string (e.g.
    # ``agents:*:*``) would grant everything here and nothing there. Refuse it so a
    # managed role can't carry a silently-divergent broad grant; the one documented
    # way to grant all actions is ``agent_os:admin``.
    if act == "*":
        raise ValueError(
            f"Action wildcard '*' is not allowed in scope {scope!r}: it would mean 'all actions' "
            f"under Casbin but nothing under the scope provider. List explicit actions "
            f"(read/run/write/delete), use a resource-id wildcard like 'agents:*:run', or grant "
            f"'agent_os:admin'."
        )
    return (obj, act)


def _obj_act_to_scope(obj: str, act: str) -> str:
    """Best-effort reverse of :func:`scope_to_obj_act`, for display/read-back.

    Lossy where two scope spellings collapse to the same policy (``agents:read``
    and ``agents:*:read`` both store as ``("agents/*", "read")``); we render the
    global ``resource:action`` form in that case.
    """
    if obj == "*":
        return "agent_os:admin"
    if obj.endswith("/*"):
        return f"{obj[:-2]}:{act}"
    resource, _, rid = obj.partition("/")
    return f"{resource}:{rid}:{act}"


class CasbinPolicyEngine(PolicyEngine):
    """Casbin-backed :class:`PolicyEngine`. Builds its own enforcer from a DB
    (``db``/``db_url``) or wraps an existing ``casbin.Enforcer``."""

    def __init__(
        self,
        db_url: Optional[str] = None,
        db: Optional[Any] = None,
        enforcer: Optional[Any] = None,
    ):
        if enforcer is not None:
            # Bring-your-own enforcer (advanced / CasbinAuthorizationProvider).
            if not hasattr(enforcer, "enforce"):
                raise TypeError(
                    "CasbinPolicyEngine requires a casbin.Enforcer "
                    f"(install with `pip install agno[roles]`). Got: {type(enforcer)!r}"
                )
            self._enforcer = enforcer
            return

        try:
            import casbin  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Managed roles need the optional extra. Install it with: pip install \"agno[roles]\""
            ) from e

        import casbin

        model = casbin.Model()
        model.load_model_from_text(_MODEL_TEXT)
        adapter_target = _engine_from_db(db) if db is not None else db_url
        if adapter_target is not None:
            from casbin_sqlalchemy_adapter import Adapter

            self._enforcer = casbin.Enforcer(model, Adapter(adapter_target))
        else:
            self._enforcer = casbin.Enforcer(model)
        self._enforcer.enable_auto_save(True)  # mutations persist immediately

    # --- authoring -------------------------------------------------------
    def set_role_scopes(self, role: str, entries: List[ScopeEntry]) -> None:
        self._enforcer.remove_filtered_policy(0, role)
        for scope, effect in entries:
            obj, act = scope_to_obj_act(scope)
            self._enforcer.add_policy(role, obj, act, effect)

    def add_scope(self, role: str, scope: str, effect: str = "allow") -> None:
        obj, act = scope_to_obj_act(scope)
        self._enforcer.remove_filtered_policy(0, role, obj, act)  # drop any prior effect
        self._enforcer.add_policy(role, obj, act, effect)

    def remove_scope(self, role: str, scope: str) -> None:
        obj, act = scope_to_obj_act(scope)
        self._enforcer.remove_filtered_policy(0, role, obj, act)

    def get_role_scopes(self, role: str) -> List[ScopeEntry]:
        out: List[ScopeEntry] = []
        for p in self._enforcer.get_filtered_policy(0, role):
            if len(p) >= 3:
                out.append((_obj_act_to_scope(p[1], p[2]), p[3] if len(p) >= 4 else "allow"))
        return out

    def remove_role(self, role: str) -> None:
        self._enforcer.remove_filtered_policy(0, role)
        self._enforcer.remove_filtered_grouping_policy(1, role)

    def list_roles(self) -> List[str]:
        return sorted({p[0] for p in self._enforcer.get_policy()})

    # --- assignments -----------------------------------------------------
    def assign(self, subject: str, role: str) -> None:
        self._enforcer.add_role_for_user(subject, role)

    def unassign(self, subject: str, role: str) -> None:
        self._enforcer.delete_role_for_user(subject, role)

    def roles_of(self, subject: str) -> List[str]:
        return list(self._enforcer.get_roles_for_user(subject))

    # --- decisions -------------------------------------------------------
    def _enforce(self, obj: str, act: str, subject: Optional[str], roles: Optional[List[str]]) -> bool:
        """One decision for ``(obj, act)``. Roles-from-token take precedence (each
        role enforced against the policy); else the subject's stored assignments."""
        if roles:
            return any(bool(self._enforcer.enforce(role, obj, act)) for role in roles)
        if not subject:
            return False
        return bool(self._enforcer.enforce(subject, obj, act))

    def check_resource(
        self,
        resource_type: Optional[str],
        resource_id: Optional[str],
        action: Optional[str],
        *,
        subject: Optional[str] = None,
        roles: Optional[List[str]] = None,
    ) -> bool:
        if not resource_type or not action:
            return True  # non-resource check: defer (the route gate handles it)
        obj = f"{resource_type}/{resource_id}" if resource_id else resource_type
        return self._enforce(obj, action, subject, roles)

    def check_scope(self, scope: str, *, subject: Optional[str] = None, roles: Optional[List[str]] = None) -> bool:
        try:
            obj, act = scope_to_obj_act(scope)
        except ValueError:
            return False  # unmappable scope -> not satisfied
        return self._enforce(obj, act, subject, roles)

    def accessible_resource_ids(
        self,
        resource_type: str,
        action: Optional[str],
        *,
        subject: Optional[str] = None,
        roles: Optional[List[str]] = None,
    ) -> Set[str]:
        """List-filtering support from the subject's implicit permissions (direct +
        via roles). ``{"*"}`` for wildcard/collection grants, else specific ids."""
        if not resource_type or not subject:
            return set()
        try:
            perms = self._enforcer.get_implicit_permissions_for_user(subject)
        except Exception:
            perms = [p for p in self._enforcer.get_policy() if p and p[0] == subject]

        ids: Set[str] = set()
        for perm in perms:
            # perm is [sub, obj, act, eft?]. Skip malformed rows; do NOT treat a
            # missing act as match-any (that was a fail-open). Skip explicit denies.
            if len(perm) < 3:
                continue
            p_obj, p_act = perm[1], perm[2]
            if p_obj is None or p_act is None:
                continue
            if len(perm) >= 4 and str(perm[3]).lower() == "deny":
                continue
            if action is not None and p_act != action and p_act != "*":
                continue
            if p_obj in ("*", f"{resource_type}/*", resource_type):
                return {"*"}
            prefix = f"{resource_type}/"
            if p_obj.startswith(prefix):
                ids.add(p_obj[len(prefix):])
        return ids
