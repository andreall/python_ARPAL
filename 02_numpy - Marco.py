# %%
import numpy as np
# prova Marco
# Commento Marco

# %%
a = np.array([10, 20.1, 30, 40])
b = list()
for n in range(5):
    b.append(n)
c = np.array(b)
c
# Marco prova

# %% [markdown]
# Numpy arrays have several attributes that give useful information about the array:

# %%
a.ndim  # number of dimensions
a.shape  # shape of the array
a.dtype  # numerical type

# %% [markdown]
# ### Generation
# NumPy also provides helper functions for generating arrays of data to save you typing for regularly spaced data. Don't forget your Python indexing rules!
# 
# * `arange(start, stop, step)` creates a range of values in the interval `[start,stop)` with `step` spacing.
# * `linspace(start, stop, num)` creates a range of `num` evenly spaced values over the range `[start,stop]`.

# %%
d =np.arange(1, 5, 1) 
# if start is not given, it will be 0!
# if step is not given, it will be 1!

# %%
np.arange(30.5)

# %%
# We can use floats
b = np.arange(1.2, 4.41, 0.1)


# %%
# LINSPACE
np.linspace(11, 12, 10)

# %% [markdown]
# and a similar function can be used to create logarithmically spaced values between and including limits:

# %%
np.logspace(1., 4., 7)

# %% [markdown]
# Finally, the ``zeros`` and ``ones`` functions can be used to create arrays intially set to ``0`` and ``1`` respectively:

# %%
np.zeros(10)

# %%
np.ones(5)

# %%
# accessing and slicing
b[0]
b[0:5:2] # Slices [start:end:step]

# %% [markdown]
# Summarizing:  
# - np.array([X, X, X, X])
# - np.arange(start, finish, step)
# - np.linspace(start, finish-included, number of elements)
# - np.zeros([dim, dim]) 1D: np.zeros(X)
# - np.ones([dim, dim])
# - np.empty([dim, dim])

# %%
a.shape

# %%
b.shape

# %%
np.vstack([a, a]).shape

# %%
np.hstack([a, a]).shape

# %%
np.stack([a, a]).shape

# %%
d = np.vstack([a, a])
d.shape

# %%
np.vstack([d, d]).shape

# %%
np.hstack([d, d]).shape

# %%
np.stack([d, d]).shape

# %%
np.dstack([d, d]).shape

# %% [markdown]
# ### Exercise
# Create an array which contains the value 2 repeated 10 times
# Create an array which contains values from 1 until 90 every one and then the values 95, 99, 99.9, 99.99.

# %% [markdown]
# ## Numerical operations with arrays
# Numpy arrays can be combined numerically using the standard ``+-*/**`` operators:


# %%
x1 = np.array([1,2,3])
y1 = np.array([4,5,6])

# %%
y1

# %%
2 * y1

# %%
(x1 + 2) * y1

# %%
x1 ** y1

# %% [markdown]
# Note that this differs from lists:

# %%
x = [1,2,3]
y = [4,5,6, 'hola']

# %%
3 * y

# %%
2 * y

# %%
x + 2 * y

# %% [markdown]
# ### Constants

# %% [markdown]
# NumPy provides us access to some useful constants as well - remember you should never be typing these in manually! Other libraries such as SciPy and MetPy have their own set of constants that are more domain specific.

# %%
np.pi

# %%
np.e

# %%
1 + np.pi

# %% [markdown]
# ### Array math functions
# 
# NumPy also has math functions that can operate on arrays. Similar to the math operations, these greatly simplify and speed up these operations. Let's start with calculating $\sin(t)$!

# %%
t = np.array([4, 6, 8])
sin_t = np.sin(t)
sin_t

# %% [markdown]
# and clean it up a bit by `rounding to three decimal places.

# %%
np.round(sin_t, 3)

# %%
cos_t = np.cos(t)
cos_t

# %% [markdown]
# <div class="admonition alert alert-info">
#     <p class="admonition-title" style="font-weight:bold">Info</p>
#     Check out NumPy's list of mathematical functions <a href=https://numpy.org/doc/stable/reference/routines.math.html>here</a>!
# </div>

# %%
degrees = np.rad2deg(t)
degrees

# %% [markdown]
# We are similarly provided algorithms for operations including integration, bulk summing, and cumulative summing.

# %%
sine_integral = np.trapz(sin_t, t)
np.round(sine_integral, 3)

# %%
cos_sum = np.sum(cos_t)
cos_sum

# %%
cos_csum = np.cumsum(cos_t)
print(cos_csum)

# %% [markdown]
# ## Multi-dimensional arrays

# %%
y = np.ones([3,2,3])  # ones takes the shape of the array, not the values

# %%
y.shape

# %%
y

# %% [markdown]
# Multi-dimensional arrays can be sliced differently along different dimensions:

# %%
y[::3, 1:4, :]

# %% [markdown]
# ## Using axes to slice arrays
# 
# Here we introduce an important concept when working with NumPy: the axis. This indicates the particular dimension along which a function should operate (provided the function does something taking multiple values and converts to a single value). 
# 
# Let's look at a concrete example with `sum`:

# %%
a = np.arange(12).reshape(3, 4)
a

# %% [markdown]
# This calculates the total of all values in the array.

# %%
np.sum(a)

# %% [markdown]
# <div class="admonition alert alert-info">
#     <p class="title" style="font-weight:bold">Info</p>
#     Some of NumPy's functions can be accessed as `ndarray` methods!
# </div>

# %%
a.sum()

# %% [markdown]
# Now, with a reminder about how our array is shaped,

# %%
a.shape

# %% [markdown]
# we can specify `axis` to get _just_ the sum across each of our rows.

# %%
np.sum(a, axis=0)

# %% [markdown]
# Or do the same and take the sum across columns:

# %%
np.sum(a, axis=1)

# %% [markdown]
# After putting together some data and introducing some more advanced calculations, let's demonstrate a multi-layered example: calculating temperature advection. If you're not familiar with this (don't worry!), we'll be looking to calculate
# 
# \begin{equation*}
# \text{advection} = -\vec{v} \cdot \nabla T
# \end{equation*}
# 
# and to do so we'll start with some random $T$ and $\vec{v}$ values,

# %%
temp = np.random.randn(100, 50)
u = np.random.randn(100, 50)
v = np.random.randn(100, 50)

# %% [markdown]
# We can calculate the `np.gradient` of our new $T(100x50)$ field as two separate component gradients,

# %%
gradient_x, gradient_y = np.gradient(temp)

# %%
gradient_x

# %% [markdown]
# In order to calculate $-\vec{v} \cdot \nabla T$, we will use `np.dstack` to turn our two separate component gradient fields into one multidimensional field containing $x$ and $y$ gradients at each of our $100x50$ points,

# %%
grad_vectors = np.dstack([gradient_x, gradient_y])
print(grad_vectors.shape)

# %% [markdown]
# and then do the same for our separate $u$ and $v$ wind components,

# %%
wind_vectors = np.dstack([u, v])
print(wind_vectors.shape)


# %% [markdown]
# Finally, we can calculate the dot product of these two multidimensional fields of wind and temperature gradient components by hand as an element-wise multiplication, `*`, and then a `sum` of our separate components at each point (i.e., along the last `axis`),

# %%
advection = (wind_vectors * -grad_vectors).sum(axis=-1)
print(advection.shape)

# %% [markdown]
# ## Masking

# %% [markdown]
# The index notation ``[...]`` is not limited to single element indexing, or multiple element slicing, but one can also pass a discrete list/array of indices:

# %%
x = np.array([1,6,4,7])

# %%
x[[True, False, True, False]]

# %% [markdown]
# which is returning a new array composed of elements 1, 2, 4, etc from the original array.

# %% [markdown]
# Alternatively, one can also pass a boolean array of ``True/False`` values, called a **mask**, indicating which items to keep:

# %%
y = np.array([3, 4, 5])

# %%
mask = np.array([True, False, False])

# %%
y[mask]

# %% [markdown]
# Now this doesn't look very useful because it is very verbose, but now consider that carrying out a comparison with the array will return such a boolean array:

# %%
x

# %%
mask = x > 3.4

# %% [markdown]
# It is therefore possible to extract subsets from an array using the following simple notation:

# %%
x[x > 3.4]

# %%
x[mask]

# %%
x[~mask]

# %% [markdown]
# Conditions can be combined:

# %% [markdown]
# ### Conditional formating
# 
# For Loops --> and, or
# 
# For masking ---> & (and), | (or)

# %%
x

# %%
x[(x > 3.4) & (x < 5.5)]

# %% [markdown]
# Of course, the boolean **mask** can be derived from a different array to ``x`` as long as it is the right size:

# %%
x = np.linspace(-1., 1., 14)
y = np.array([1,6.,4,7,9.,3,1,5,6,7,3,4,4,3])

# %%
y.shape

# %%
x.shape

# %%
y2 = y + 3
y2

# %%
mask = y2 >= 9
yy = y[mask]
yy

# %%
y[(x > -0.5) | (x < 0.4)]

# %% [markdown]
# Since the mask itself is an array, it can be stored in a variable and used as a mask for different arrays:

# %%
keep = (x > -0.5) & (x < 0.4)
x_new = x[keep]
y_new = y[keep]

# %%
keep

# %%
x_new

# %%
y_new

# %% [markdown]
# we can use this conditional indexing to assign new values to certain positions within our array, somewhat like a masking operation.

# %%
y

# %%
mask = y>5
mask

# %%
y[mask] = 999

# %%
y

# %%
y1 = y[mask]
y1

# %%
mm = y > 3

# %%
mm

# %%
y[mm] = np.NaN

# %%
y

# %%
y[y > 5] = 3

# %% [markdown]
# ### NaN values

# %% [markdown]
# In arrays, some of the values are sometimes NaN - meaning *Not a Number*. If you multiply a NaN value by another value, you get NaN, and if there are any NaN values in a summation, the total result will be NaN. One way to get around this is to use ``np.nansum`` instead of ``np.sum`` in order to find the sum:

# %%
x = np.array([1,2,3,np.NaN])
x

# %%
np.NAN # np.nan | np.NaN | np.NAN

# %%
np.nansum(x)

# %%
np.nansum(x)

# %%
np.nanmax(x)

# %% [markdown]
# You can also use ``np.isnan`` to tell you where values are NaN. For example, ``array[~np.isnan(array)]`` will return all the values that are not NaN (because ~ means 'not'):

# %%
np.isnan(x)

# %%
x[np.isnan(x)]

# %%
x[~np.isnan(x)]

# %% [markdown]
# ## Exercise

# %% [markdown]
# The [data/SIMAR_gaps.txt](data/SIMAR_gaps.txt) data file gives the wave climate data in the Mediterranean Sea.

# %% [markdown]
# Read in the file using ``np.loadtxt``. The data contains bad values, which you can identify by looking at the minimum and maximum values of the array. Use masking to get rid of the bad values.

# %%
from pathlib import Path
dir_data = Path('data')
data = np.loadtxt(dir_data / 'SIMAR_gaps.txt', skiprows=1)
#np.genfromtxt()
var = data[:, 4]

# %% [markdown]
# ## Using broadcasting to implicitly loop over data

# %% [markdown]
# ### What is broadcasting?
# Broadcasting is a useful NumPy tool that allows us to perform operations between arrays with different shapes, provided that they are compatible with each other in certain ways. To start, we can create an array below and add 5 to it:

# %%
a = np.array([10, 20, 30, 40])
a + 5  # works with a number

# %%
b = np.array([5])
a + b # works with an array

# %% [markdown]
# This takes the single element in `b` and adds it to each of the elements in `a`. This won't work for just any `b`, though; for instance, the following won't work:

# %%
b = np.array([5, 6, 7])
# a + b
# It does work if `a` and `b` are the same shape:

# %%
b = np.array([5, 5, 10, 10])
a + b

# %% [markdown]
# What if what we really want is pairwise addition of a and b? Without broadcasting, we could accomplish this by looping:

# %%
b = np.array([1, 2, 3, 4, 5])

# %%
a

# %%
b

# %%
result = np.empty((5, 4), dtype=np.int32)
for row, valb in enumerate(b):
    for col, vala in enumerate(a):
        result[row, col] = vala + valb
result

# %% [markdown]
# ### Giving NumPy room for broadcasting
# We can also do this using broadcasting, which is where NumPy implicitly repeats the array without using additional memory. With broadcasting, NumPy takes care of repeating for you, provided dimensions are "compatible". This works as follows:
# 1. Check the number of dimensions of the arrays. If they are different, *prepend* dimensions of size one until the arrays are the same dimension shape.
# 2. Check if each of the dimensions are compatible. This works as follows:
#   - Each dimension is checked.
#   - If one of the arrays has a size of 1 in the checked dimension, or both arrays have the same size in the checked dimension, the check passes.
#   - If all dimension checks pass, the dimensions are compatible.
# 
# For example, consider the following arrays:

# %%
a.shape

# %%
b.shape

# %% [markdown]
# Right now, these arrays both have the same number of dimensions.  They both have only one dimension, but that dimension is incompatible.  We can solve this by appending a dimension using `np.newaxis` when indexing, like this:

# %%
bb = b[:, np.newaxis]
bb.shape

# %%
bb

# %%
a + bb

# %%
(a + bb).shape

# %% [markdown]
# We can also make the code more succinct by performing the newaxis and addition operations in a single line, like this:

# %%
a + b[:, np.newaxis]

# %% [markdown]
# ### Extending to higher dimensions
# The same broadcasting ability and rules also apply for arrays of higher dimensions. Consider the following arrays `x`, `y`, and `z`, which are all different dimensions. We can use newaxis and broadcasting to perform $x^2 + y^2 + z^2$:

# %%
x = np.array([1, 2])
y = np.array([3, 4, 5])
z = np.array([6, 7, 8, 9])

# %% [markdown]
# First, we extend the `x` array using newaxis, and then square it.  Then, we square `y`, and broadcast it onto the extended `x` array:

# %%
d_2d = x[:, np.newaxis] ** 2 + y**2

# %%
d_2d.shape

# %% [markdown]
# Finally, we further extend this new 2-D array to a 3-D array using newaxis, square the `z` array, and then broadcast `z` onto the newly extended array:

# %%
d_3d = d_2d[..., np.newaxis] + z**2

# %%
d_3d.shape

# %% [markdown]
# As described above, we can also perform these operations in a single line of code, like this:

# %%
h = x[:, np.newaxis, np.newaxis] ** 2 + y[np.newaxis, :, np.newaxis] ** 2 + z**2

# %% [markdown]
# Given the 3-D temperature field and 1-D pressure coordinates below, let's calculate $T * exp(P / 1000)$. We will need to use broadcasting to make the arrays compatible.  The following code demonstrates how to use newaxis and broadcasting to perform this calculation:

# %%
pressure = np.array([1000, 850, 500, 300])
temps = np.linspace(20, 30, 24).reshape(4, 3, 2)


# %% [markdown]
# ## Vectorize calculations to avoid explicit loops

# %% [markdown]
# When working with arrays of data, loops over the individual array elements is a fact of life. However, for improved runtime performance, it is important to avoid performing these loops in Python as much as possible, and let NumPy handle the looping for you. Avoiding these loops frequently, but not always, results in shorter and clearer code as well.

# %% [markdown]
# ### Look ahead/behind
# 
# One common pattern for vectorizing is in converting loops that work over the current point, in addition to the previous point and/or the next point. This comes up when doing finite-difference calculations, e.g., approximating derivatives:
# 
# \begin{equation*}
# f'(x) = f_{i+1} - f_{i}
# \end{equation*}

# %%
a = np.linspace(0, 20, 6)
a

# %% [markdown]
# We can calculate the forward difference for this array using a manual loop, like this:

# %%
d = np.zeros(a.size - 1)
for i in range(len(a) - 1):
    d[i] = a[i + 1] - a[i]
d

# %% [markdown]
# It would be nice to express this calculation without a loop, if possible. To see how to go about this, let's consider the values that are involved in calculating `d[i]`; in other words, the values `a[i+1]` and `a[i]`. The values over the loop iterations are:
# 
# |  i  | a[i+1] | a[i] |
# | --- |  ----  | ---- |
# |  0  |    4   |   0  |
# |  1  |    8   |   4  |
# |  2  |   12   |   8  |
# |  3  |   16   |  12  |
# |  4  |   20   |  16  |
# 
# We can then express the series of values for `a[i+1]` as follows:

# %%
a[1:]

# %% [markdown]
# We can also express the series of values for `a[i]` as follows:

# %%
a[:-1]

# %% [markdown]
# This means that we can express the forward difference using the following statement:

# %%
a[1:] - a[:-1]

# %% [markdown]
# #### 2nd Derivative
#     
# A finite-difference estimate of the 2nd derivative is given by the following equation (ignoring $\Delta x$):
# 
# \begin{equation*}
# f''(x) = 2
# f_i - f_{i+1} - f_{i-1}
# \end{equation*}
# 
# Let's write some vectorized code to calculate this finite difference for `a`, using slices.  Analyze the code below, and compare the result to the values you would expect to see from the 2nd derivative of `a`.


