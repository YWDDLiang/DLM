"""Fixed-denominator metric identity for raw and complete-system endpoints."""

PAPER_METRICS = (
    "composition_valid",
    "structure_valid",
    "Direct_joint",
    "novel",
    "unique",
    "novel_unique",
    "CHGNet_energy_per_atom",
    "official_e_above_hull",
    "Strict_SUN",
    "Meta_SUN",
)

UNKNOWN_POLICY = "retain in requested denominator; never map stable"

__all__ = ["PAPER_METRICS", "UNKNOWN_POLICY"]
