"""Render the public paper result without component-level internal tables."""

from __future__ import annotations

from .science import PUBLIC_RESULT


def main() -> None:
    strict = PUBLIC_RESULT["strict_sun"]
    meta = PUBLIC_RESULT["meta_sun"]
    print("H1-A2 method family")
    print(
        f"Strict S.U.N.: {strict['numerator']}/{strict['denominator']} "
        f"= {100 * strict['rate']:.2f}%"
    )
    print(
        f"Meta S.U.N.: {meta['numerator']}/{meta['denominator']} "
        f"= {100 * meta['rate']:.2f}%"
    )


if __name__ == "__main__":
    main()

