'''
Epsilon lower bound computation adopted from Jie Zhang's implementation of Steinke 2023.
https://github.com/zj-jayzhang/one_round_auditing/blob/main/src/dp_audit.py
'''

import math
import scipy.stats

def p_value_DP_audit(m, r, v, eps, delta):
        """
        Calculate the p-value of achieving >=v correct guesses under the null hypothesis.

        Args:
            m: Number of examples, each included independently with probability 0.5.
            r: Number of guesses (i.e., excluding abstentions).
            v: Number of correct guesses by the auditor.
            eps: Epsilon, DP guarantee of null hypothesis.
            delta: Delta, DP guarantee of null hypothesis.

        Returns:
            p-value: Probability of >=v correct guesses under the null hypothesis.
        """
        assert 0 <= v <= r <= m
        assert eps >= 0
        assert 0 <= delta <= 1
        # import pdb; pdb.set_trace()
        q = 1 / (1 + math.exp(-eps))  # Accuracy of eps-DP randomized response
        beta = scipy.stats.binom.sf(v - 1, r, q)  # P[Binomial(r, q) >= v]
        alpha = 0
        total_sum = 0  # P[v > Binomial(r, q) >= v - i]

        for i in range(1, v + 1):
            total_sum += scipy.stats.binom.pmf(v - i, r, q)
            if total_sum > i * alpha:
                alpha = total_sum / i

        p = beta + alpha * delta * 2 * m
        return min(p, 1)

# Function to compute the lower bound on epsilon
def get_eps_audit(m, r, v, delta, p):
    """
    Compute the lower bound on epsilon such that the algorithm is not (eps, delta)-DP.

    Args:
        m: Number of examples, each included independently with probability 0.5.
        r: Number of guesses (i.e., excluding abstentions).
        v: Number of correct guesses by the auditor.
        delta: Delta, DP guarantee of null hypothesis.
        p: 1 - confidence (e.g., p=0.05 corresponds to 95%).

    Returns:
        eps_min: Lower bound on epsilon.
    """
    assert 0 <= v <= r <= m
    assert 0 <= delta <= 1
    assert 0 < p < 1

    eps_min = 0  # p_value_DP_audit(eps_min) < p
    eps_max = 1  # p_value_DP_audit(eps_max) >= p
    
    # Expand eps_max until p_value_DP_audit(eps_max) >= p
    while p_value_DP_audit(m, r, v, eps_max, delta) < p:
        eps_max += 1

    # Binary search to find eps_min
    for _ in range(30):
        eps = (eps_min + eps_max) / 2
        if p_value_DP_audit(m, r, v, eps, delta) < p:
            eps_min = eps
        else:
            eps_max = eps

    return eps_min