import csv
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def norm_vec(X, y, reg_ransac):
    reg_ransac.fit(X, y)
    normal_vector = reg_ransac.estimator_.coef_
    return normal_vector

def is_plane(X, y, normal_vector, threshold=0.1):
    distances = np.abs(y - (X @ normal_vector))
    avg = np.mean(distances)
    if avg < threshold:
        return True
    else:
        return False

def is_vertical(normal_vector):
    vertical = np.isclose(normal_vector, [0, 0, 1]).all() or np.isclose(normal_vector, [0, 0, -1]).all()
    return vertical


def load_data_from_file():
    with open('dane_cwiczenie2.xyz', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            x, y, z = map(float, row)
            yield x, y, z


if __name__ == '__main__':
    chmury = list(load_data_from_file())
    alg_kmeans = KMeans(n_clusters=3)
    cluster = alg_kmeans.fit_predict(chmury)
    chart = plt.figure()
    chart3D = chart.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b']

    for i in range(3):
        tablica = np.array(chmury)[cluster == i]
        Y = tablica
        ransac = RANSACRegressor()
        normal_vector = norm_vec(tablica, Y, ransac)
        check_plane = is_plane(tablica, Y, normal_vector)
        check_orient = is_vertical(normal_vector)
        print("Wyniki dla chmury ", i+1, ':')
        print("Wektor normalny:", normal_vector)
        print("Czy chmura jest płaszczyzną:")
        if check_plane:
            print("Tak")
            print("Czy płaszczyzna jest pionowa:")
            if check_orient:
                print("Tak")
            else:
                print("Nie")
        else:
            print("Nie")
        chart3D.scatter(tablica[:, 0], tablica[:, 1], tablica[:, 2], c=colors[i], label=f'Przyporzadkowanie {i + 1}')
    plt.legend()
    plt.show()