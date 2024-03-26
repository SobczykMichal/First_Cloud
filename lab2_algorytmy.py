import csv
from sklearn.cluster import KMeans
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
import numpy as np
def load_data_from_file():
    with open('dane_cwiczenie2.xyz', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            x, y, z = map(float, row)
            yield x, y, z

chmury = list(load_data_from_file())
kmeans = KMeans(n_clusters=3)
cluster = kmeans.fit_predict(chmury)
wykres = plt.figure()
chart3D = wykres.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b']
for i in range(3):
    tablica = np.array(chmury)[cluster == i]
    reg = RANSACRegressor().fit(tablica[:, :2], tablica[:, 2:3])
    prostopadly=np.append(reg.estimator_.coef_,0)
    print(f'chmura {i + 1}:')
    print('wektor prostopadly do plaszczyzny:', prostopadly)
    chart3D.scatter(tablica[:, 0], tablica[:, 1], tablica[:, 2], c=colors[i], label=f'Przyporzadkowanie {i + 1}')
plt.legend()
plt.show()