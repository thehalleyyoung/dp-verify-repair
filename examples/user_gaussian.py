# Example: Gaussian mechanism for (epsilon, delta)-differential privacy.
# Verify with:  dpcegar verify examples/user_gaussian.py --budget "eps=1.0,delta=1e-5" --notion approx
# Check with:   dpcegar check examples/user_gaussian.py

# @dp.mechanism(privacy="(1.0, 1e-5)-dp", sensitivity=1.0)
def mean_query(db, query):
    """Gaussian mechanism for answering a mean query."""
    # @dp.sensitivity(1.0)
    true_answer = query(db)
    # @dp.noise(kind="gaussian", sigma=7.07)
    noise = gaussian(0, 7.07)
    result = true_answer + noise
    return result
