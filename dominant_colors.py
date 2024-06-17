import cv2
import numpy as np
import argparse

def display_colors_with_opencv(centers):
    swatch_size = 200
    swatches_image = np.zeros((swatch_size, len(centers) * swatch_size, 3), dtype=np.uint8)

    for i, color in enumerate(centers):
        swatch_start = i * swatch_size
        swatch_end = (i + 1) * swatch_size
        swatches_image[:, swatch_start:swatch_end, :] = color

        text_position = (swatch_start + 5, swatch_size - 10)
        cv2.putText(swatches_image, str(tuple(color[::-1])), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imshow("Dominant Colors", swatches_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(img_path, color_count):
    img = cv2.imread(img_path)

    height, width, channels = img.shape
    print("Image size:", width, "x", height)

    data = np.reshape(img, (width * height, channels))
    data = np.float32(data)

    num_clusters = color_count
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data, num_clusters, None, criteria, 10, flags)

    centers = np.uint8(centers)
    print("Dominant Colors (RGB format):")
    for color in centers:
        rgb_color = tuple(color[::-1])  # BGR to RGB
        print(rgb_color)

    display_colors_with_opencv(centers)

parser = argparse.ArgumentParser(description='Find dominant colors on image with K-Means Clustering algorithm.')
parser.add_argument('--img_path', type=str, help='Image path', required=True)
parser.add_argument('--color_count', type=int, help='Number of dominant colors to display', required=False, default=3)
args = parser.parse_args()

img_path = args.img_path
color_count = args.color_count
main(img_path, color_count)