import numpy as np
import scipy.ndimage
import sys

N_images = 618
D1 = np.zeros((N_images,N_images),dtype=np.float)
D2 = np.zeros((N_images,N_images),dtype=np.float)
for i in range(N_images):
    sys.stdout.write("\rComputing distances between images: %d/%d"%(i+1, N_images))
    sys.stdout.flush()
    I1 = scipy.ndimage.imread("pendulum_video/pendulum_%04d.png"%(i)).reshape(-1).astype(np.float)
    for j in range(i+1,N_images):
        I2 = scipy.ndimage.imread("pendulum_video/pendulum_%04d.png"%(j)).reshape(-1).astype(np.float)
        D1[i,j] = D1[j,i] = np.linalg.norm(I1-I2, ord=1)
        D2[i,j] = D2[j,i] = np.linalg.norm(I1-I2, ord=2)


l1_fname = "D1.npy"
l2_fname = "D2.npy"
np.save(l1_fname, D1)
np.save(l2_fname, D2)
sys.stdout.write("\nDistances based on l1 and l2 norms saved to %s and %s\n"%(l1_fname, l2_fname))
