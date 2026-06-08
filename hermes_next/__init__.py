"""Hermes Next — Next-generation memory provider for Hermes Agent."""

from hermes_next.provider import HermesNextProvider

__version__ = "0.3.0"
__all__ = ["HermesNextProvider", "register"]


def register():
    """Plugin entry point — returns the provider class for Hermes Agent discovery."""
    return HermesNextProvider
