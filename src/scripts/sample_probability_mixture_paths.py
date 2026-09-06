#!/usr/bin/env python3
"""Explicit entry for the fixed equal same-state K4/K8 probability mixture."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.sample_state_programmed_paths import main

if __name__=="__main__":
    if not any(arg=="--mixture-peer-checkpoint" or arg.startswith("--mixture-peer-checkpoint=") for arg in sys.argv[1:]) and "--help" not in sys.argv:
        raise SystemExit("this entry requires --mixture-peer-checkpoint with K8; --checkpoint-path is K4")
    main()
