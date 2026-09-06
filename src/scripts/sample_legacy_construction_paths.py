#!/usr/bin/env python3
"""Explicit construction-only entry for an original legacy K4/K8 policy."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sample_state_programmed_paths import main

if __name__=="__main__":
    if "--construction-only" not in sys.argv:sys.argv.append("--construction-only")
    main()
