import numpy as np

arrays = [np.random.randn(3, 4) for _ in range(10)]
print np.stack(arrays, axis=0).shape