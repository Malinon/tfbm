from scipy.special import gamma
import mpmath


def gamma_inc(a, z):
    return mpmath.gammainc(z=a, a=0, b=z)

def mittag_leffer(H, z):
    if z == 0:
        # 1 / gamma(beta)
        return 1.0 / gamma(3 - 2 * H)
    if H == 0.5:
        # Extreme case H = 0.5
        return 1.0
    a1 = 1 - 2 * H
    a2 = 2 - 2 * H
    numerator = -(z * gamma_inc(a1, -z) + gamma_inc(a2, -z))
    denominator = gamma(a1) * (-z)**(a2)
    return numerator / denominator
