"""Optional Casbin-backed AuthorizationProvider (bring-your-own-enforcer).

For advanced users who already have a configured ``casbin.Enforcer`` (their own
model + DB adapter) and want to plug it straight into AgentOS. It's now a thin
wrapper: the Casbin logic lives in
:class:`~agno.os.authz.casbin_engine.CasbinPolicyEngine` (the swappable backend),
and the generic :class:`~agno.os.authz.engine.EngineAuthorizationProvider` turns
any engine into an :class:`AuthorizationProvider`.

OPT-IN. Casbin is not a hard dependency; install with ``pip install agno[roles]``.
The default provider remains the zero-dependency
:class:`~agno.os.authz.scope_provider.ScopeAuthorizationProvider`.

Usage::

    import casbin
    from agno.os.authz.casbin_provider import CasbinAuthorizationProvider
    from agno.os.config import AuthorizationConfig

    enforcer = casbin.Enforcer("model.conf", adapter)   # adapter -> your DB
    agent_os = AgentOS(
        agents=[...],
        authorization=True,
        authorization_config=AuthorizationConfig(
            verification_keys=[PUBLIC_KEY],
            authorization_provider=CasbinAuthorizationProvider(enforcer),
        ),
    )

Policy convention: ``sub`` = principal, ``obj`` = ``"<resource_type>/<resource_id>"``
(or ``"<resource_type>"`` for collections; use ``keyMatch2`` for ``agents/*``),
``act`` = action. Role assignment uses Casbin grouping (``g, user, role``).
"""

from typing import Any, Optional

# Re-exported for back-compat: scope_to_obj_act now lives with the Casbin engine.
from agno.os.authz.casbin_engine import CasbinPolicyEngine, scope_to_obj_act  # noqa: F401
from agno.os.authz.engine import EngineAuthorizationProvider


class CasbinAuthorizationProvider(EngineAuthorizationProvider):
    """Authorization decisions delegated to a Casbin enforcer you supply."""

    def __init__(self, enforcer: Any, roles_claim: Optional[str] = None):
        """
        Args:
            enforcer: a configured ``casbin.Enforcer`` (sync). You own the model +
                adapter, so policy storage stays in your infra.
            roles_claim: optional JWT claim carrying the caller's roles (e.g.
                ``"roles"`` for Auth0/WorkOS). When set and present, roles come FROM
                the token and each is authorized against the policy (the IdP case);
                otherwise roles come from the enforcer's ``g`` assignments keyed on
                the subject (the no-IdP case).
        """
        super().__init__(CasbinPolicyEngine(enforcer=enforcer), roles_claim=roles_claim)
