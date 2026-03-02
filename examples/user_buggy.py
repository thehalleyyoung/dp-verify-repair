# Example: Buggy Laplace mechanism (wrong noise scale).
# DP-CEGAR should DETECT this bug.
# Verify with:  dpcegar verify examples/user_buggy.py --budget "eps=1.0"
#
# BUG: scale should be sensitivity/epsilon = 1.0, but is set to 0.5
# (insufficient noise → privacy violation).

# @dp.mechanism(privacy="1.0-dp", sensitivity=1.0)
def buggy_counting(db, query):
    """Laplace mechanism with wrong noise scale (too little noise)."""
    # @dp.sensitivity(1.0)
    true_answer = query(db)
    # @dp.noise(kind="laplace", scale=0.5)
    noise = laplace(0, 0.5)
    result = true_answer + noise
    return result
