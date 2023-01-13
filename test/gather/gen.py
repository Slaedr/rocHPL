
import numpy as np
import numpy.random as npr
import numpy.linalg as nlin

bs = 80
n_block_dim = 15
N = bs * n_block_dim

A = np.random.rand(N, N)
b = np.random.rand(N)

for i in range(n_block_dim):
    for j in range(n_block_dim):
        ab = A[i*bs:(i+1)*bs, j*bs:(j+1)*bs]
        fname = "A_{}_{}.npy".format(i,j)
        np.save(fname, ab)

np.save('full_A.npy', A)
np.save('b.npy', b)

fp = open('desc.txt', 'w')
fp.write("grid_a grid_b N N_rhs\n")
fp.write("{} {} {} {}".format(n_block_dim, n_block_dim, bs*n_block_dim, 1));
fp.close()
