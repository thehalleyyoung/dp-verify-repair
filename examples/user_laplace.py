# Example: Correct Laplace mechanism.
# Verify with:  dpcegar verify examples/user_laplace.py --budget "eps=1.0"
# Check with:   dpcegar check examples/user_laplace.py

# @dp.mechanism(privacy="1.0-dp", sensitivity=1.0)
def counting_query(db, query):
    """Standard Laplace mechanism for a counting query (sensitivity=1)."""
    # @dp.sensitivity(1.0)
    true_answer = query(db)
    # @dp.noise(kind="laplace", scale=1.0)
    noise = laplace(0, 1.0)
    result = true_answer + noise
    return result
