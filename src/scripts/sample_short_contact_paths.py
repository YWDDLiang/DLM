#!/usr/bin/env python3
"""Explicit entry for the optional fixed K4/K8 tilt in the original sampler."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sample_state_programmed_paths import main

if __name__ == "__main__":
    if not any(arg == "--short-contact-spec" or arg.startswith("--short-contact-spec=") for arg in sys.argv[1:]) and "--help" not in sys.argv:
        raise SystemExit("this entry requires --short-contact-spec; use the original entry for an unmodified policy")
    main()
