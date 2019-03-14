
import numpy as np


a = np.array([(1,2,3,4,5), (4,5,6,7,8)])
np.save('arr.npy', a)
print(np.load('arr.npy'))