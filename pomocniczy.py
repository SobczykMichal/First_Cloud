from skimage.feature import graycomatrix, graycoprops
import cv2
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import numpy as np
from PIL import Image
PATCH_SIZE = 21

# Wczytaj obraz za pomocą OpenCV
img = cv2.imread('0_1152.png')
Image.fromarray(img)
while(True):
    pass
"""
if image is None:
    print("Błąd: Nie można wczytać obrazu.")
    exit()
cv2.imshow("Okno obrazek . jpg ",image)


# Wybierz fragmenty z obszarów trawiastych na obrazie
grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
grass_patches = []
for loc in grass_locations:
    grass_patch = image[loc[0] : loc[0] + PATCH_SIZE, loc[1] : loc[1] + PATCH_SIZE]
    grass_patches.append(grass_patch)

# Wybierz fragmenty z obszarów niebieskiego nieba na obrazie
sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
sky_patches = []
for loc in sky_locations:
    sky_patch = image[loc[0] : loc[0] + PATCH_SIZE, loc[1] : loc[1] + PATCH_SIZE]
    sky_patches.append(sky_patch)

# Oblicz niektóre właściwości GLCM dla każdego fragmentu
xs = []
ys = []
for patch in grass_patches + sky_patches:
    glcm = graycomatrix(
        patch, distances=[5], angles=[0], levels=256, symmetric=True, normed=True
    )
    xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(graycoprops(glcm, 'correlation')[0, 0])

# Pozostała część kodu pozostaje taka sama...

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=255)
for y, x in grass_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
for y, x in sky_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[: len(grass_patches)], ys[: len(grass_patches)], 'go', label='Grass')
ax.plot(xs[len(grass_patches) :], ys[len(grass_patches) :], 'bo', label='Sky')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()

# display the image patches
for i, patch in enumerate(grass_patches):
    ax = fig.add_subplot(3, len(grass_patches), len(grass_patches) * 1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255)
    ax.set_xlabel(f"Grass {i + 1}")

for i, patch in enumerate(sky_patches):
    ax = fig.add_subplot(3, len(sky_patches), len(sky_patches) * 2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255)
    ax.set_xlabel(f"Sky {i + 1}")


# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()
"""