# SPDX-License-Identifier: Apache-2.0
"""Kit loads `Extension` from here. Outside Isaac Sim (tests, headless runs) only the
pure-Python fleet modules are importable, so the omni import is guarded."""
try:
    import omni.ext  # noqa: F401

    from .extension import *  # noqa: F401,F403
except ImportError:  # running outside Isaac Sim
    pass
