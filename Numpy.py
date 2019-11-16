import numpy as np
a=np.array([1,2,3])

print(a)
print(a.ndim)
print(a.itemsize)
print(a.dtype)
print(a.size)
print(a.shape)
print(type(a))



b=np.array([(1,2,3),(2,4,6)])
print(b)
print(b.ndim)
print(b.itemsize)
print(b.dtype)
print(b.size)
print(b.shape)
print(type(b))

print(b.reshape(3,2))
print(b[0,2])
print(b[0:,2])

print(b.sum(axis=1))
print(np.sqrt(b))
print(np.std(b))

# +,-,*,/ operation on two numpy array

# vertical and horizontal stacking for numpy array
x=np.array([2,4,6])
y=np.array([6,8,10])

print(np.vstack((x,y)))
print(np.hstack((x,y)))
z=np.vstack((x,y))
print(z)
print(z.ravel())

#numpy special function - sin,arange,cos,exp,log,log10

