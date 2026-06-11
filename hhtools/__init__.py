"""hhtools - Human-to-Humanoid Tools.

A unified Python toolkit for importing human motion capture, retargeting to humanoid robots,
and analyzing motion datasets.

All user-facing types live in submodules; importing the top-level package only pulls in the
lightweight core API so that CLI startup is fast.
"""

from __future__ import annotations

from hhtools._version import __version__

__all__ = ["__version__"]
