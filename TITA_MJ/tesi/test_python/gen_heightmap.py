import os

import numpy as np
import imageio

np.random.seed(42)

nrow, ncol = 200, 200  # risoluzione alta
radius_hole = 0.1      # raggio del buco centrale in metri
max_height = 0.008       # altezza massima sassolini ±7 cm

# coordinate normalizzate da -1 a 1
x = np.linspace(-1, 1, nrow)
y = np.linspace(-1, 1, ncol)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# terreno inizialmente piatto
terrain = np.zeros((nrow, ncol))

# Sassolini solo fuori dal cerchio centrale
num_rocks = 5000

# genera posizioni casuali per i sassolini fuori dal cerchio centrale
positions = []
while len(positions) < num_rocks:
    xi, yi = np.random.randint(0, nrow), np.random.randint(0, ncol)
    if R[xi, yi] > radius_hole:
        positions.append((xi, yi))

# disegna ciascun sassolino come un piccolo picco gaussiano, capped a max_height
for xi, yi in positions:
    sigma = 1.0   # dimensione del sassolino in pixel
    th_range = int(3*sigma)
    for i in range(-th_range, th_range + 1):
        for j in range(-th_range, th_range + 1):
            if 0 <= xi+i < nrow and 0 <= yi+j < ncol:
                terrain[xi+i, yi+j] = np.maximum(
                    terrain[xi+i, yi+j],
                    max_height * np.exp(-(i**2 + j**2)/(2*sigma**2))
                )
# salva come PNG
terrain_img = ((terrain - terrain.min()) / (terrain.max() - terrain.min()) * 255).astype(np.uint8)
#imageio.imwrite("heightmap.png", terrain_img)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "heightmap.png")

imageio.imwrite(output_path, terrain_img)