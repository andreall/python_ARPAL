
# Prova Laura 20231016-2
#%%
u = 1
v = 4

from utils import uv2mag
Uw = uv2mag(u, v)

# %%
import numpy as np
u2 = np.arange(1, 10)
v2 = np.arange(5, 14)
Uw2 = uv2mag(u2, v2)

# %%
### prova Patti