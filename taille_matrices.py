from config import*
import numpy as np


gcd = np.gcd(alpha * beta, L)
p = alpha * beta // gcd
q = L // gcd
print(f"alpha*beta / L = {p}/{q}")





