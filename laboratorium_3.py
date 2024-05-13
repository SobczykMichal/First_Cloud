import os
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import csv

def extract_texture_samples(input_folder, output_folder, sample_size):
    # Tworzenie folderu wyjściowego, jeśli nie istnieje
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lista wszystkich plików w folderze wejściowym
    files = os.listdir(input_folder)

    # Iteracja po wszystkich plikach
    for file in files:
        # Odczyt obrazu
        image_path = os.path.join(input_folder, file)
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Nie udało się wczytać obrazu: {file}")
            print(e)
            continue

        # Wycięcie fragmentów tekstury o zadanym rozmiarze
        width, height = image.size
        for i in range(0, height - sample_size + 1, sample_size):
            for j in range(0, width - sample_size + 1, sample_size):
                box = (j, i, j + sample_size, i + sample_size)
                sample = image.crop(box)

                # Tworzenie folderu dla danej tekstury, jeśli nie istnieje
                category_folder = os.path.join(output_folder, file.split('.')[0])
                if not os.path.exists(category_folder):
                    os.makedirs(category_folder)

                # Zapisanie wyciętego fragmentu do odpowiedniego folderu
                sample.save(os.path.join(category_folder, f"{i}_{j}.png"))

    print("Zakończono wycinanie fragmentów tekstury.")

def extract_texture_features(input_folder1,input_folder2,input_folder3, output_file, sample_size):
    # Lista cech tekstury
    features = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']

    # Przyjęte odległości pikseli i kierunki
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0, 45, 90, 135 stopni

    # Otwarcie pliku CSV do zapisu
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Zapisanie nagłówka
        header = features + ['category']
        writer.writerow(header)

        # Iteracja po wszystkich plikach
        for category in os.listdir(input_folder1):
            category_folder = os.path.join(input_folder1, category)
            if not os.path.isdir(category_folder):
                continue

            # Iteracja po wszystkich plikach w kategorii
            for file in os.listdir(category_folder):
                image_path = os.path.join(category_folder, file)
                try:
                    image = Image.open(image_path).convert('L')  # Konwersja do skali szarości
                    image = image.quantize(64)  # Zmniejszenie głębi jasności do 5 bitów (64 poziomy)
                except Exception as e:
                    print(f"Nie udało się wczytać obrazu: {file}")
                    print(e)
                    continue

                # Obliczanie cech tekstury dla każdego fragmentu
                glcm = graycomatrix(np.array(image), distances=distances, angles=angles, symmetric=True, normed=True)
                texture_features = np.hstack([graycoprops(glcm, prop).ravel() for prop in features])

                # Dodanie nazwy kategorii
                texture_features = np.append(texture_features, category)

                # Zapisanie cech do pliku CSV
                writer.writerow(texture_features)
            print("Zapisano cechy tekstury1 do pliku CSV.")
        for category in os.listdir(input_folder2):
            category_folder = os.path.join(input_folder2, category)
            if not os.path.isdir(category_folder):
                continue

            # Iteracja po wszystkich plikach w kategorii
            for file in os.listdir(category_folder):
                image_path = os.path.join(category_folder, file)
                try:
                    image = Image.open(image_path).convert('L')  # Konwersja do skali szarości
                    image = image.quantize(64)  # Zmniejszenie głębi jasności do 5 bitów (64 poziomy)
                except Exception as e:
                    print(f"Nie udało się wczytać obrazu: {file}")
                    print(e)
                    continue

                # Obliczanie cech tekstury dla każdego fragmentu
                glcm = graycomatrix(np.array(image), distances=distances, angles=angles, symmetric=True, normed=True)
                texture_features = np.hstack([graycoprops(glcm, prop).ravel() for prop in features])
                # Dodanie nazwy kategorii
                texture_features = np.append(texture_features, category)
                # Zapisanie cech do pliku CSV
                writer.writerow(texture_features)
            print("Zapisano cechy tekstury2 do pliku CSV.")
        for category in os.listdir(input_folder3):
            category_folder = os.path.join(input_folder3, category)
            if not os.path.isdir(category_folder):
                continue

            # Iteracja po wszystkich plikach w kategorii
            for file in os.listdir(category_folder):
                image_path = os.path.join(category_folder, file)
                try:
                    image = Image.open(image_path).convert('L')  # Konwersja do skali szarości
                    image = image.quantize(64)  # Zmniejszenie głębi jasności do 5 bitów (64 poziomy)
                except Exception as e:
                    print(f"Nie udało się wczytać obrazu: {file}")
                    print(e)
                    continue

                # Obliczanie cech tekstury dla każdego fragmentu
                glcm = graycomatrix(np.array(image), distances=distances, angles=angles, symmetric=True, normed=True)
                texture_features = np.hstack([graycoprops(glcm, prop).ravel() for prop in features])
                # Dodanie nazwy kategorii
                texture_features = np.append(texture_features, category)
                # Zapisanie cech do pliku CSV
                writer.writerow(texture_features)
            print("Zapisano cechy tekstury3 do pliku CSV.")

    print("Zapisano cechy tekstury do pliku CSV.")

# Użycie funkcji
input_folder1 = r"C:\Users\Nitro\Pictures\LAB_POI\desk"
output_folder1 = r"C:\Users\Nitro\Pictures\LAB_POI\128format_desk"
input_folder2 = r"C:\Users\Nitro\Pictures\LAB_POI\curtains"
output_folder2 = r"C:\Users\Nitro\Pictures\LAB_POI\128format_curtains"
input_folder3 = r"C:\Users\Nitro\Pictures\LAB_POI\tiles"
output_folder3 = r"C:\Users\Nitro\Pictures\LAB_POI\128format_tiles"
gray_file_out1=r"C:\Users\Nitro\Pictures\LAB_POI\gray_desk.csv"
gray_file_out2=r"C:\Users\Nitro\Pictures\LAB_POI\gray_curtains.csv"
gray_file_out3=r"C:\Users\Nitro\Pictures\LAB_POI\gray_tiles.csv"
gray_file_out=r"C:\Users\Nitro\Pictures\LAB_POI\gray_all.csv"
gray_file_in1=r"C:\Users\Nitro\Pictures\LAB_POI\128format_desk"
gray_file_in2=r"C:\Users\Nitro\Pictures\LAB_POI\128format_curtains"
gray_file_in3=r"C:\Users\Nitro\Pictures\LAB_POI\128format_tiles"
sample_size = 128

extract_texture_samples(input_folder1, output_folder1, sample_size)
extract_texture_samples(input_folder2, output_folder2, sample_size)
extract_texture_samples(input_folder3, output_folder3, sample_size)

#extract_texture_features(output_folder1, gray_file_out1, sample_size)
#extract_texture_features(output_folder2, gray_file_out2, sample_size)
#extract_texture_features(output_folder3, gray_file_out3, sample_size)

extract_texture_features(output_folder3,output_folder2,output_folder1, gray_file_out, sample_size)
