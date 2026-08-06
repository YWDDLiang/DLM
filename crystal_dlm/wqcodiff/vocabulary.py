"""Frozen MP20 species and space-group vocabulary helpers."""

from __future__ import annotations


# MP20 contains Z=1..83 and Z=89..94.  Po/At/Rn/Fr/Ra are absent.  Class IDs
# are contiguous so all matched methods expose exactly the same 89-way head.
MP20_ATOMIC_NUMBERS = tuple(range(1, 84)) + tuple(range(89, 95))
ATOMIC_NUMBER_TO_CLASS = {
    atomic_number: index for index, atomic_number in enumerate(MP20_ATOMIC_NUMBERS)
}


def atomic_number_to_input_id(atomic_number: int) -> int:
    """Return 1..89; zero is the registered MASK token."""

    try:
        return ATOMIC_NUMBER_TO_CLASS[int(atomic_number)] + 1
    except KeyError as exc:
        raise ValueError(
            f"atomic number {atomic_number} is outside the frozen MP20 vocabulary"
        ) from exc


def atomic_number_to_target(atomic_number: int) -> int:
    return atomic_number_to_input_id(atomic_number) - 1


def target_to_atomic_number(target: int) -> int:
    if not 0 <= int(target) < len(MP20_ATOMIC_NUMBERS):
        raise ValueError(f"species target {target} is outside [0,88]")
    return MP20_ATOMIC_NUMBERS[int(target)]


def crystal_system_from_space_group(space_group: int) -> str:
    """Return the International Tables crystal system for SG 1--230."""

    value = int(space_group)
    if not 1 <= value <= 230:
        raise ValueError("space group must be in [1,230]")
    if value <= 2:
        return "triclinic"
    if value <= 15:
        return "monoclinic"
    if value <= 74:
        return "orthorhombic"
    if value <= 142:
        return "tetragonal"
    if value <= 167:
        return "trigonal"
    if value <= 194:
        return "hexagonal"
    return "cubic"

