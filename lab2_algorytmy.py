import csv
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def load_data_from_file():
    #load data
    with open('dane_cwiczenie2.xyz', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            x, y, z = map(float, row)
            yield x, y, z


def Kmeans_method(points, n_clusters=3):
    #Kmeans method, set clusters, show method
    alg_kmeans = KMeans(n_clusters=n_clusters)
    cluster = alg_kmeans.fit_predict(points)
    chart = plt.figure()
    plt.title('KMeans method')
    chart3D = chart.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b']
    for i in range(n_clusters):
        matrix_points = np.array(points)[cluster == i]
        chart3D.scatter(matrix_points[:, 0], matrix_points[:, 1], matrix_points[:, 2], c=colors[i], label=f'Przyporzadkowanie {i + 1}')
        plane_other, inlier_mask_other = fit_plane_ransac_sklearn(matrix_points)

        print("Wektor normalny do płaszczyzny ", i+1, ":", plane_other[:3])
        avg = avg_distance(plane_other, matrix_points)
        print("Średnia odległość punktów od płaszczyzny:", avg)
        if (np.linalg.norm(plane_other[:3]) < 5):
            print("Chmura tworzy plaszczyzne")
        else:
            print("Chmura nie tworzy plaszczyzny")
        normalized_vector = plane_other[:3]/np.linalg.norm(plane_other[:3])
        if np.abs(normalized_vector[0]) < 0.1:
            if np.abs(normalized_vector[1]) < 0.1:
                print("Płaszczyzna jest pozioma")
            else:
                print("Płaszczyzna jest pionowa")
        else:
            print("Płaszczyzna o innej orientacji")


def fit_plane_ransac_sklearn(points, max_trials=100):
     X = points[:, :2]  # Współrzędne x i y
     y = points[:, 2]  # Współrzędna z
        # RANSAC method
     ransac = RANSACRegressor(max_trials=max_trials)
     ransac.fit(X, y)

     # plane
     a, b = ransac.estimator_.coef_
     c = ransac.estimator_.intercept_

     return np.array([a, b, -1, c]), ransac.inlier_mask_  # matrix of plane [a, b, -1, c] for ax+by-z+c=0


def plot_RANSAC(plane, points, inlier_mask):
    a, b, _, c = plane

    inliers = points[inlier_mask]
    outliers = points[~inlier_mask]

    chart = plt.figure()
    plt.title('RANSAC method')
    chart3D = chart.add_subplot(111, projection='3d')

    chart3D.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2], c='b', label='Inliers')
    chart3D.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2], c='r', label='Outliers')


def avg_distance(plane, points):
    a, b, _, c = plane
    distances = np.abs(a * points[:, 0] + b * points[:, 1] - points[:, 2] + c) / np.sqrt(a**2 + b**2 + (-1)**2)
    return np.mean(distances)


if __name__ == '__main__':

    cloud_points = list(load_data_from_file())
    tablica = np.array(cloud_points)
    plane_from_clouds, inlier_Points = fit_plane_ransac_sklearn(tablica)
    plot_RANSAC(plane_from_clouds, tablica, inlier_Points)

    print("Wektor normalny do calkowitej płaszczyzny:", plane_from_clouds[:3])
    mean_distance = avg_distance(plane_from_clouds, tablica)
    print("Średnia odległość punktów od płaszczyzny:", mean_distance)

    Kmeans_method(cloud_points, 3)

    plt.legend()
    plt.show()