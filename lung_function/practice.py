import numpy as np


volume =np.random.randint(1024, size=(512, 512))
volume = volume.astype(np.uint16)
print(volume.dtype)

a = (volume - (volume.min()+0))
print(a.dtype)
print('max of volume-min_value:{}'.format(np.max(a)))
b = (volume - (volume.min()+1))
print(b.dtype)

print('max of volume-min_value2:{}'.format(np.max(b)))
