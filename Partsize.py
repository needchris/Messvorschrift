import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture

# Sicherstellen, dass der Ordner 'images' existiert
if not os.path.exists('images'):
    os.makedirs('images')

# Bildpfad definieren
images_folder_path = './data/'
probe_number = "5022"
image_files = os.listdir(images_folder_path)
selected_file = next((file for file in image_files if f"Probe{probe_number}" in file), None)

if not selected_file:
    raise FileNotFoundError(f"Keine Bilddatei für Probe {probe_number} gefunden.")

image_path = os.path.join(images_folder_path, selected_file)
print(f"Analysiere Bild: {selected_file}")

# Bild laden
image_to_show = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Bild visualisieren
plt.imshow(image_to_show)
# plt.show()

# Konvertierung in den HSV-Farbraum
hsv_image = cv2.cvtColor(image_to_show, cv2.COLOR_BGR2HSV)
hue_channel, saturation_channel, value_channel = cv2.split(hsv_image)

# Kanäle anzeigen
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(hue_channel, cmap='hsv')
ax1.set_title('Hue Channel')

ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(saturation_channel, cmap='gray')
ax2.set_title('Saturation Channel')

ax3 = fig.add_subplot(1, 3, 3)
ax3.imshow(value_channel, cmap='gray')
ax3.set_title('Value Channel')

# plt.show()

# Otsu-Thresholding auf den Value-Kanal anwenden
otsu_threshold, bg = cv2.threshold(value_channel, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
mask = cv2.threshold(value_channel, otsu_threshold, 255, cv2.THRESH_BINARY)[1]

plt.imshow(mask, cmap='gray')
# plt.show()

# Adaptive Thresholding auf alle Kanäle anwenden
hue_mask = cv2.adaptiveThreshold(hue_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
saturation_mask = cv2.adaptiveThreshold(saturation_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
value_mask = cv2.adaptiveThreshold(value_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

# Ergebnisse visualisieren
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1, 4, 1)
ax1.imshow(cv2.cvtColor(image_to_show, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image')

ax2 = fig.add_subplot(1, 4, 2)
ax2.imshow(hue_mask, cmap='gray')
ax2.set_title('Hue Mask')

ax3 = fig.add_subplot(1, 4, 3)
ax3.imshow(saturation_mask, cmap='gray')
ax3.set_title('Saturation Mask')

ax4 = fig.add_subplot(1, 4, 4)
ax4.imshow(value_mask, cmap='gray')
ax4.set_title('Value Mask')

# plt.show()

# Konturen finden
contours, _ = cv2.findContours(value_mask + saturation_mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
filtered_contours = [contour for contour in contours
                     if 100 * 500 < cv2.contourArea(contour) < image_to_show.size * 0.9]

print(f"Gefundene Konturen: {len(filtered_contours)}")

# Größte Kontur markieren
if filtered_contours:
    largest_contour = max(filtered_contours, key=cv2.contourArea)
    cv2.drawContours(image_to_show, [largest_contour], 0, (0, 255, 0), 5)

plt.imshow(cv2.cvtColor(image_to_show, cv2.COLOR_BGR2RGB))
plt.show()

# Eckenpunkte bestimmen
height, width = image_to_show.shape[:2]
top_left, top_right = np.array([0, 0]), np.array([width - 1, 0])
bottom_left, bottom_right = np.array([0, height - 1]), np.array([width - 1, height - 1])

reshaped_contour = largest_contour.reshape(-1, 1, 2)

def find_closest_point(contour, corner):
    distances = np.sqrt(((contour - corner) ** 2).sum(axis=2))
    return tuple(contour[np.argmin(distances)][0])

closest_top_left = find_closest_point(reshaped_contour, top_left)
closest_top_right = find_closest_point(reshaped_contour, top_right)
closest_bottom_left = find_closest_point(reshaped_contour, bottom_left)
closest_bottom_right = find_closest_point(reshaped_contour, bottom_right)

for point in [closest_top_left, closest_top_right, closest_bottom_left, closest_bottom_right]:
    cv2.circle(image_to_show, point, 50, (0, 255, 0), -1)

plt.imshow(image_to_show)
plt.show()

# Transformation anwenden
transformation_matrix = cv2.getPerspectiveTransform(
    np.float32([closest_top_left, closest_top_right, closest_bottom_left, closest_bottom_right]),
    np.float32([top_left, top_right, bottom_left, bottom_right])
)

rectified_image = cv2.warpPerspective(image_to_show, transformation_matrix, (width, height))
rectified_image = cv2.resize(rectified_image, (1500, 3000))

plt.imshow(cv2.cvtColor(rectified_image, cv2.COLOR_BGR2RGB))
plt.show()

# Bereich definieren
roi_top_left, roi_bottom_right = (650, 258), (845, 2700)
area_of_interest = rectified_image[roi_top_left[1]:roi_bottom_right[1],
                                   roi_top_left[0]:roi_bottom_right[0]]

plt.imshow(area_of_interest)
plt.show()

# Konvertierung in den HSV-Farbraum für den Bereich von Interesse
area_of_interest_hsv = cv2.cvtColor(area_of_interest, cv2.COLOR_BGR2HSV)
v_channel = area_of_interest_hsv[:, :, 2]

# GMM-Analyse auf den V-Kanal anwenden
v_channel_data = v_channel.reshape(-1, 1)

# Gaussian Mixture Model mit zwei Komponenten (Hintergrund und Streifen)
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(v_channel_data)

# Vorhersage der Labels
labels = gmm.predict(v_channel_data)
segmented_image = labels.reshape(v_channel.shape)

# Bestimmen, welches Label den Streifen entspricht
if gmm.means_[0] < gmm.means_[1]:
    # Label 0 ist Hintergrund
    binary_image = np.where(segmented_image == 1, 255, 0).astype(np.uint8)
else:
    # Label 1 ist Hintergrund
    binary_image = np.where(segmented_image == 0, 255, 0).astype(np.uint8)

# Segmentiertes Bild anzeigen mit Mikrometerskala und Rahmen
plt.figure(figsize=(8, 8))
plt.imshow(binary_image, cmap='gray')
plt.title('200% Probe')
##################################################################################
# Mikrometerskala hinzufügen
height = binary_image.shape[0]
positions_pixels = np.linspace(0, height, num=6)
positions_mm = positions_pixels * (145.0 / height)
particle_sizes_um = 50 * (1 - positions_mm / 145.0)

plt.gca().set_ylim([height, 0])  # Y-Achse invertieren
plt.gca().yaxis.set_ticks(positions_pixels)
plt.gca().yaxis.set_ticklabels([f'{int(size)} µm' for size in particle_sizes_um])
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position("right")

# Entfernen der X-Achse und der unteren Skala
plt.gca().xaxis.set_visible(False)

# Entfernen der "Pixel"-Beschriftungen
plt.gca().yaxis.set_label_text('')  # Y-Achsen-Beschriftung entfernen

# Bild speichern
gmm_segmentation_filename = f'images/GMM_Segmentierung_Probe{probe_number}.png'
plt.savefig(gmm_segmentation_filename,dpi=600, bbox_inches='tight')
plt.show()

# Bild vertikal spiegeln, falls notwendig
binary_image_flipped = binary_image  # Passen Sie dies an, falls erforderlich

# Bestimmen der Skalierungsfaktoren
height, width = binary_image_flipped.shape
mm_per_pixel = 145.0 / height  # ROI ist 145 mm hoch

# Definieren der Slice-Höhe in Pixeln (1 mm pro Slice)
slice_height_mm = 1  # mm
slice_height_pixels = max(int(slice_height_mm / mm_per_pixel), 1)  # Mindestens 1 Pixel

num_slices = int(height / slice_height_pixels)

# Mindestdicke für Streifen in Pixeln
min_streak_width = 10  # Pixel

# Initialisieren der Liste für die Anzahl der Streifen pro Slice
num_streaks_per_slice = []

# Durchlaufen der Slices und Zählen der Streifen
for i in range(num_slices):
    y_start = i * slice_height_pixels
    y_end = min(y_start + slice_height_pixels, height)
    slice_img = binary_image_flipped[y_start:y_end, :]

    # Sicherstellen, dass das Slice zweidimensional ist
    if slice_img.ndim == 2 and slice_img.shape[0] > 0 and slice_img.shape[1] > 0:
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(slice_img)
        num_streaks = 0  # Zähler für gültige Streifen

        for j in range(1, num_labels):  # Start bei 1, da 0 der Hintergrund ist
            width_comp = stats[j, cv2.CC_STAT_WIDTH]
            height_comp = stats[j, cv2.CC_STAT_HEIGHT]
            # Überprüfen, ob die Streifenbreite und -höhe größer als der Schwellenwert sind
            if width_comp >= min_streak_width or height_comp >= min_streak_width:
                num_streaks += 1
        num_streaks_per_slice.append(num_streaks)
    else:
        num_streaks_per_slice.append(0)

# Funktion zur Umrechnung von Position in mm zu Partikelgröße in µm
def position_to_particle_size(position_mm):
    return 50 * (1 - position_mm / 145)

# Visualisierung der Streifenverteilung mit Seaborn
positions_mm = np.array([i * slice_height_mm for i in range(num_slices)])
particle_sizes_um = position_to_particle_size(positions_mm)

sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.5)
plt.figure(figsize=(10, 6))
sns.lineplot(x=particle_sizes_um, y=num_streaks_per_slice, marker='o', color='navy', linewidth=2.5)
plt.xlabel('Partikelgröße (µm)')
plt.ylabel('Anzahl der Streifen')
#plt.title('Anzahl der Streifen pro Partikelgröße')
plt.gca().invert_xaxis()  # X-Achse invertieren, um 50 µm links zu haben
plt.grid(True, linestyle='--', linewidth=0.5)

# Achsenbeschriftungen und Ticks anpassen
plt.xticks(np.arange(0, 55, 5))
plt.xlim(50, 0)

# Finden der Position, an der der erste Streifen erscheint (maximale Partikelgröße)
max_particle_size_um = None
max_particle_idx = None
for idx, num_streaks in enumerate(num_streaks_per_slice):
    if num_streaks > 0:
        position_mm = idx * slice_height_mm
        particle_size_um = position_to_particle_size(position_mm)
        print(f"Erster gültiger Streifen erscheint bei Position {position_mm:.2f} mm, Partikelgröße: {particle_size_um:.2f} µm")
        max_particle_size_um = particle_size_um
        max_particle_idx = idx
        break
else:
    print("Keine gültigen Streifen im Bild gefunden.")

# Markierung der maximalen Partikelgröße im Plot und Hinzufügen der Mikrometerwerte
if max_particle_size_um is not None:
    plt.axvline(x=max_particle_size_um, color='red', linestyle='--', label='Maximale Partikelgröße')
    plt.text(max_particle_size_um + 1, max(num_streaks_per_slice)*0.8, f'{max_particle_size_um:.1f} µm',
             color='red', rotation=90, va='center')

# Finden des ersten 3 mm Intervalls mit 5-10 gültigen Streifen (mittlere Partikelgröße nach DIN)
mean_particle_size_um = None
interval_mm = 3  # mm
interval_slices = int(interval_mm / slice_height_mm)

if max_particle_idx is not None:
    for idx in range(max_particle_idx, len(num_streaks_per_slice) - interval_slices + 1):
        total_streaks = sum(num_streaks_per_slice[idx:idx + interval_slices])
        if 5 <= total_streaks <= 10:
            position_mm = idx * slice_height_mm
            particle_size_um = position_to_particle_size(position_mm)
            print(
                f"Mittlere Partikelgröße nach DIN-Norm bei Position {position_mm:.2f} mm, Partikelgröße: {particle_size_um:.2f} µm")
            mean_particle_size_um = particle_size_um
            # Markierung der mittleren Partikelgröße im Plot und Hinzufügen der Mikrometerwerte
            plt.axvline(x=mean_particle_size_um, color='green', linestyle='-.', label='Mahlfeinheit')
            plt.text(mean_particle_size_um + 1, max(num_streaks_per_slice)*0.6, f'{mean_particle_size_um:.1f} µm',
                     color='green', rotation=90, va='center')
            break
    else:
        print("Kein Intervall gefunden, in dem 5-10 gültige Streifen vorkommen.")
else:
    print("Maximale Partikelgröße nicht gefunden, kann mittlere Partikelgröße nicht bestimmen.")

plt.legend()
plt.tight_layout()

# Diagramm speichern
streak_diagram_filename = f'images/Streifenverteilung_Probe{probe_number}.png'
plt.savefig(streak_diagram_filename,dpi=1000, bbox_inches='tight')
plt.show()

# Ergebnisse ausgeben
if max_particle_size_um is not None:
    print(f"Maximale Partikelgröße: {max_particle_size_um:.2f} µm")
if mean_particle_size_um is not None:
    print(f"Mittlere Partikelgröße (DIN-Norm): {mean_particle_size_um:.2f} µm")
