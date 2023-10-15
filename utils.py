import numpy as np

def uv2mag(u, v):
    mag = np.sqrt(u**2 + v**2)

    return mag
