#!/usr/bin/env python3
"""Thin wrapper — delegates to hermes_next.migrate.main().

Usage:
    python scripts/migrate_memos.py --old-db ~/.hermes/memos-plugin/data/memos.db
"""

from __future__ import annotations

import sys

if __name__ == "__main__":
    from hermes_next.migrate import main
    sys.exit(main())
