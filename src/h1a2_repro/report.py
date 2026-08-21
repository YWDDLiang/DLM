"""Render the public paper result without component-level internal tables."""

from __future__ import annotations

from .science import PUBLIC_RESULT


def main() -> None:
    strict = PUBLIC_RESULT["strict_sun"]
    meta = PUBLIC_RESULT["meta_sun"]
    print(PUBLIC_RESULT["method"])
    print(f"Primary view: {PUBLIC_RESULT['primary_view']}")
    print(
        f"Strict S.U.N.: {strict['numerator']}/{strict['denominator']} "
        f"= {100 * strict['rate']:.2f}%"
    )
    print(
        f"Meta S.U.N.: {meta['numerator']}/{meta['denominator']} "
        f"= {100 * meta['rate']:.2f}%"
    )
    exact = PUBLIC_RESULT["exact_all_attempt_view"]
    print("Exact all-requested-attempt audit view:")
    print(
        f"  Strict {exact['strict_sun']['numerator']}/{exact['strict_sun']['denominator']}"
        f"; Meta {exact['meta_sun']['numerator']}/{exact['meta_sun']['denominator']}"
    )
    compat = PUBLIC_RESULT["historical_compatibility_view"]
    print("Historical compatibility view (not primary):")
    print(
        f"  Strict {compat['strict_sun']['numerator']}/{compat['strict_sun']['denominator']}"
        f"; Meta {compat['meta_sun']['numerator']}/{compat['meta_sun']['denominator']}"
    )


if __name__ == "__main__":
    main()
