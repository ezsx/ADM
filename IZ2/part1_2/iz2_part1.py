import cv2
from utils.utils_scripts import clear_output_folder, show_plot


def only_gaussian_blur(image_path, blur_size):
    image = cv2.imread(image_path, 0)  # Read the image in grayscale
    blurred_image = cv2.GaussianBlur(image, blur_size, 0)

    # in case you want to see the plot
    # show_plot(blurred_image, f"Blur: {blur_size}")

    # save image to output folder
    cv2.imwrite(f"output/{blur_size}.png", blurred_image)


def apply_canny(image_path, blur_size, lower_threshold, upper_threshold):
    image = cv2.imread(image_path, 0)  # Read the image in grayscale
    blurred_image = cv2.GaussianBlur(image, blur_size, 0)
    edges = cv2.Canny(blurred_image, lower_threshold, upper_threshold)

    # in case you want to see the plot
    # show_plot(edges, f"Blur: {blur_size}, Thresholds: {lower_threshold}, {upper_threshold}")

    # save image to output folder
    cv2.imwrite(f"output/{blur_size}_{lower_threshold}_{upper_threshold}.png", edges)


image_path = r"D:\ADM\IZ2\images\img2.png"  # Replace with your image path

# Example: Apply Canny with different Gaussian blur parameters and thresholds
blur_sizes = [(3, 3), (5, 5), (7, 7)]
thresholds = [(50, 150), (100, 200), (150, 250)]

clear_output_folder()
for blur_size in blur_sizes:
    for lower_threshold, upper_threshold in thresholds:
        apply_canny(image_path, blur_size, lower_threshold, upper_threshold)
        print(f"Gauss Blur: {blur_size}, Thresholds: {lower_threshold}, {upper_threshold}")
