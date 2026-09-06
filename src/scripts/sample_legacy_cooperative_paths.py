#!/usr/bin/env python3
"""Explicit construct-plus-cooperative endpoint of an original legacy policy."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sample_state_programmed_paths import main

if __name__=="__main__":
    if "--cooperative-only" not in sys.argv:sys.argv.append("--cooperative-only")
    main()
