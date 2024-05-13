from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Wczytanie danych z pliku CSV
data = pd.read_csv(r"C:\Users\Nitro\Pictures\LAB_POI\gray_all.csv")

# Podział danych na cechy (X) i etykiety (y)
X = data.drop(columns=['category'])
y = data['category']

# Podział danych na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicjalizacja klasyfikatora k-najbliższych sąsiadów
knn = KNeighborsClassifier(n_neighbors=5)

# Trenowanie klasyfikatora na zbiorze treningowym
knn.fit(X_train, y_train)

# Przewidywanie klas dla zbioru testowego
y_pred = knn.predict(X_test)

# Obliczenie dokładności klasyfikacji
accuracy = accuracy_score(y_test, y_pred)

# Wyświetlenie dokładności klasyfikacji
print("Dokładność klasyfikacji:", accuracy)
