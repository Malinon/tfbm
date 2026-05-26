import mpmath

# Set high precision to handle the large terms that nearly cancel
mpmath.mp.dps = 100

# --- Term 1 ---
# -2*Gamma[0.75]*(1-HypergeometricPFQ[{1, -0.5}, {0.25, 0.5, 1}, 625]) / Gamma[0.25]
hyp1 = mpmath.hyper([1, -0.5], [0.25, 0.5, 1], 625)
term1 = (-2 * mpmath.gamma(0.75) * (1 - hyp1)) / mpmath.gamma(0.25)

# --- Term 2 ---
# HypergeometricPFQ[{1, 0.25}, {1, 1.75, 1.25}, 625] * (50^1.5)*Gamma[0.25] / (Gamma[1.25]* 0.75 * (2^1.5))
hyp2 = mpmath.hyper([1, 0.25], [1, 1.75, 1.25], 625)
num2 = hyp2 * (50**1.5) * mpmath.gamma(0.25)
den2 = mpmath.gamma(1.25) * 0.75 * (2**1.5)
term2 = num2 / den2

# --- Final Expression ---
# (term1 + term2) / sqrt[pi]
result = (term1 + term2) / mpmath.sqrt(mpmath.pi)

print(f"Result: {result}")
print("Part1 term:", term1 / mpmath.sqrt(mpmath.pi))
print("Part2 term:", term2 / mpmath.sqrt(mpmath.pi))