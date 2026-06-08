"""Future adapter boundary for real Factorio server integration."""

from __future__ import annotations

from dataclasses import dataclass



@dataclass
class FactorioApiClient:
    """Placeholder client for a Factorio headless server adapter."""

    host: str = "127.0.0.1"
    port: int = 34197

    def connect(self) -> None:
        """Connect to a future Factorio control interface."""
        # TODO: Integrate with a Factorio headless server via RCON or a mod bridge.
        raise NotImplementedError("Real Factorio integration is not implemented yet.")

    def send_action(self, action: str) -> None:
        """Send an action to the future Factorio adapter."""
        # TODO: Translate agent actions into Factorio commands once the adapter exists.
        raise NotImplementedError("Real Factorio integration is not implemented yet.")

    def get_observation(self) -> dict[str, object]:
        """Read an observation from the future Factorio adapter."""
        # TODO: Read player, inventory, entity, and objective state from Factorio.
        raise NotImplementedError("Real Factorio integration is not implemented yet.")
