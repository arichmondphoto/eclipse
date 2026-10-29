"""Core processing library for the Solar Eclipse Toolkit.

Everything in this package works on plain Python objects and NumPy arrays
and never calls Siril directly, so it can be tested on its own and reused
outside Siril. The Siril-facing pieces live in ``siril/``.
"""

__version__ = "0.1.0"

from .frames import FrameMeta  # noqa: F401
