"""DP-CEGAR: Differential Privacy CEGAR Verification and Repair Engine.

This package provides a complete pipeline for verifying and repairing
differential privacy mechanisms using Counter-Example Guided Abstraction
Refinement (CEGAR).

Modules:
    ir        - Intermediate representation (types, nodes, visitors)
    parser    - Front-end parsers for mechanism description languages
    paths     - Path enumeration and symbolic execution
    density   - Density function lifting and manipulation
    smt       - SMT encoding and solver interface
    cegar     - CEGAR loop orchestration
    repair    - Automated repair strategies
    variants  - DP variant support (zCDP, RDP, f-DP, GDP)
    certificates - Proof certificate generation
    cli       - Command-line interface
    utils     - Shared utilities (logging, config, errors, timing, math)
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
