import cv2
import numpy as np
from utils.utils_scripts import clear_output_folder, show_plot


def calculate_gradient_scharr(image):
    grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    magnitude = cv2.magnitude(grad_x, grad_y)
    angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)
    return magnitude, angle


def calculate_gradient_prewitt(image):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    grad_x = cv2.filter2D(image, cv2.CV_64F, kernelx)
    grad_y = cv2.filter2D(image, cv2.CV_64F, kernely)

    # Convert to same type before magnitude calculation
    grad_x = np.float32(grad_x)
    grad_y = np.float32(grad_y)

    magnitude = cv2.magnitude(grad_x, grad_y)
    angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)
    return magnitude, angle


def calculate_gradient_laplacian(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return np.abs(laplacian), None  # Laplacian does not have a specific direction


def non_maximum_suppression(magnitude, angle):
    height, width = magnitude.shape
    output = np.zeros((height, width), dtype=np.float32)

    if angle is None:  # In case of Laplacian operator where angle is not defined
        return magnitude  # Skip non-maximum suppression

    # Angle normalization between 0 and 180 degrees
    angle[angle < 0] += 180

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            q = 255
            r = 255

            # Angle quantization to 4 directions (0, 45, 90, 135 degrees)
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                output[i, j] = magnitude[i, j]

    return output


def double_thresholding(image, low_threshold_ratio=0.05, high_threshold_ratio=0.15):
    high_threshold = image.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio

    strong_i, strong_j = np.where(image >= high_threshold)
    zeros_i, zeros_j = np.where(image < low_threshold)
    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))

    image[strong_i, strong_j] = 255
    image[zeros_i, zeros_j] = 0
    image[weak_i, weak_j] = 75  # Weak edges marked with gray color (value 75)

    return image


def edge_tracking_by_hysteresis(image):
    height, width = image.shape
    strong = 255
    weak = 75

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if image[i, j] == weak:
                # Check if one of the neighbors is strong
                if ((image[i + 1, j - 1] == strong) or (image[i + 1, j] == strong) or (image[i + 1, j + 1] == strong)
                        or (image[i, j - 1] == strong) or (image[i, j + 1] == strong)
                        or (image[i - 1, j - 1] == strong) or (image[i - 1, j] == strong) or (
                                image[i - 1, j + 1] == strong)):
                    image[i, j] = strong
                else:
                    image[i, j] = 0

    return image


def custom_edge_detection(image_path, blur_size, low_threshold_ratio, high_threshold_ratio, operator):
    image = cv2.imread(image_path, 0)
    blurred_image = cv2.GaussianBlur(image, blur_size, 0)

    if operator == 'Scharr':
        magnitude, angle = calculate_gradient_scharr(blurred_image)
    elif operator == 'Prewitt':
        magnitude, angle = calculate_gradient_prewitt(blurred_image)
    elif operator == 'Laplacian':
        magnitude, angle = calculate_gradient_laplacian(blurred_image)

    suppressed = non_maximum_suppression(magnitude, angle)
    thresholded = double_thresholding(suppressed, low_threshold_ratio, high_threshold_ratio)
    edges = edge_tracking_by_hysteresis(thresholded)

    # in case you want to see the plot
    # cv2.imshow('Edges', edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # save image to output folder
    cv2.imwrite(f"output/{operator}_{blur_size}_{low_threshold_ratio}_{high_threshold_ratio}.png", edges)


image_path = r"D:\ADM\IZ2\images\img1.png"  # Replace with your image path

# Example usage
custom_edge_detection(image_path, (5, 5), 0.05, 0.15, 'Scharr')

# operators = ['Scharr', 'Prewitt', 'Laplacian']
operators = ['Laplacian']
blur_sizes = [(3, 3), (5, 5), (7, 7)]
thresholds = [(0.05, 0.15), (0.1, 0.2), (0.15, 0.25)]

# clear_output_folder()
progress_bar = 0
for operator in operators:
    for blur_size in blur_sizes:
        for lower_threshold, upper_threshold in thresholds:
            custom_edge_detection(image_path, blur_size, lower_threshold, upper_threshold, operator)
            progress_bar += 1
            print(f'Progress: {progress_bar}/{len(operators) * len(blur_sizes) * len(thresholds)}')
