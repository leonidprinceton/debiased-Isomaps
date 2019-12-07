import numpy as np
import matplotlib.pyplot as plt
import sys
import sklearn.linear_model
import sklearn.utils.graph
import sklearn.decomposition

DISTANCE = "D1"
SUBTRACT_BIAS = True
THRESHOLD_OUT_OF_MAX = 0.5

distance_file = DISTANCE + ".npy"

try:
    D = np.load(distance_file)
except IOError:
    print("Distance file not found (%s). Try running compute_distances.py."%(distance_file))
    sys.exit(1)

print("Checking if metric is consistent and computing its bias.")
nearest_distances = np.diagonal(D[1:,:-1])
second_nearest_distances = np.diagonal(D[2:,:-2])
sum_of_nearest_distances = nearest_distances[1:] + nearest_distances[:-1]

plt.subplot(221)
plt.scatter(second_nearest_distances, sum_of_nearest_distances, color="b", marker=".", label="data")

#fit a line via RANdom SAmple Consensus
ransac = sklearn.linear_model.RANSACRegressor()
ransac.fit(second_nearest_distances.reshape(-1, 1), sum_of_nearest_distances)
distances_extent = np.array([[0], [np.max(second_nearest_distances)]])
ransac_extent_fit = ransac.predict(distances_extent)
line_slope = (ransac_extent_fit[1]-ransac_extent_fit[0])/(distances_extent[1] - distances_extent[0])
bias = ransac_extent_fit[0]
print("Slope of metric consistensy test is %.2f (should be close to 1)"%(line_slope))
print("Metrix bias estimate is %.2e"%(bias))

plt.plot(distances_extent, ransac_extent_fit, color="k", label="linear fit")
plt.plot(distances_extent, distances_extent, color="r", ls="--", label="slope of 1")
plt.ticklabel_format(style="sci", scilimits=(0,0))
plt.ylim((0,None))
plt.xlabel("D[i,i+2]")
plt.ylabel("D[i,i+1] + D[i+1,i+2]")
plt.legend(loc=2)

if SUBTRACT_BIAS:
    print("Substracting bias from all distances.")
    debiased_D = (D-bias)*(D>bias)
else:
    debiased_D = D

threshold = THRESHOLD_OUT_OF_MAX*np.max(D)

print("Discarding distances above %.2e and running Isomaps"%(threshold))
isomaps_D = debiased_D*(debiased_D<threshold)
geodesic_D = sklearn.utils.graph.graph_shortest_path(isomaps_D)
kernel_pca = sklearn.decomposition.KernelPCA(n_components=2, kernel="precomputed")
isomaps_coordinates = kernel_pca.fit_transform(-0.5*geodesic_D**2)

x = isomaps_coordinates[:,0]
y = isomaps_coordinates[:,1]

plt.subplot(222)
plt.title("Isomaps fit")
plt.plot(x, y, color="k")

print("Mapping Isomaps coordinates to angle")
center_x,center_y,_ = np.linalg.solve([[np.sum(x**2),np.sum(x*y),np.sum(x)], [np.sum(x*y), np.sum(y**2),np.sum(y)], [np.sum(x),np.sum(y),len(x)]], [0.5*np.sum(x*(x**2 + y**2)), 0.5*np.sum(y*(x**2 + y**2)), 0.5*np.sum(x**2 + y**2)])
angles = np.unwrap(np.arctan2(y-center_y, x-center_x))

plt.scatter([center_x], [center_y], marker="x", color="r")
plt.ticklabel_format(style="sci", scilimits=(0,0))

plt.subplot(223)
plt.title("Resulting angles")
plt.plot(np.degrees(angles), color="k")
plt.xlabel("frame number")
plt.ylabel("angle [deg]")

plt.subplot(224)
plt.title("Resulting angle increments")
plt.plot(np.degrees(angles[1:]-angles[:-1]), color="k")
plt.xlabel("frame number")
plt.ylabel("angle [deg]")

plt.tight_layout()
plt.show()
