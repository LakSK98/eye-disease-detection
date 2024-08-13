import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis
from skimage.filters import sobel
from save_processed_images import save_images_and_get_urls

files_directory = '/content/drive/My Drive/eye_data/Dataset/'

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def measure_opacity(image, mask=None):
    if mask is not None:
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return np.mean(masked_image[mask == 255])
    return np.mean(image)

def create_central_mask(image):
    h, w = image.shape
    mask = np.zeros_like(image, dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = int(min(h, w) * 0.35)
    cv2.circle(mask, center, radius, 255, -1)
    return mask

def remove_small_objects(binary_image, min_size):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    output_image = np.zeros(binary_image.shape, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output_image[labels == i] = 255
    return output_image

def extract_blood_vessels(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_opening)
    thresh = remove_small_objects(thresh, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    morphology = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Apply mask to the original image
    return morphology

def calculate_vessel_area(image):
    vessels = extract_blood_vessels(image)
    vessel_area = np.sum(vessels == 255)
    total_area = vessels.size
    return vessel_area / total_area

def extract_color_features(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    channels = cv2.split(image)
    mean_blue, median_blue, std_blue, brightness_blue = np.mean(channels[0]), np.median(channels[0]), np.std(channels[0]), np.mean(channels[0])
    mean_green, median_green, std_green, brightness_green = np.mean(channels[1]), np.median(channels[1]), np.std(channels[1]), np.mean(channels[1])
    mean_red, median_red, std_red, brightness_red = np.mean(channels[2]), np.median(channels[2]), np.std(channels[2]), np.mean(channels[2])
    return mean_blue, median_blue, std_blue, brightness_blue, mean_green, median_green, std_green, brightness_green, mean_red, median_red, std_red, brightness_red

def extract_texture_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    entropy = shannon_entropy(image)
    return contrast, correlation, energy, homogeneity, entropy

def extract_shape_features(image):
    edges = cv2.Canny(image, 100, 200)
    edge_density = np.sum(edges) / edges.size
    return edge_density

def extract_gradient_features(image):
    gradient = sobel(image)
    mean_gradient = np.mean(gradient)
    std_gradient = np.std(gradient)
    return mean_gradient, std_gradient

def extract_histogram_features(image):
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    skewness = skew(histogram)
    kurt = kurtosis(histogram)
    return skewness, kurt

def extract_lbp_features(image):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    return lbp_hist.flatten()

def extract_features(image):
    color_img = image
    img = preprocess_image(image)
    _, background_mask = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY_INV)
    mask = create_central_mask(img)
    central_opacity = measure_opacity(img, mask)
    overall_opacity = measure_opacity(img)
    mean_blue, median_blue, std_blue, brightness_blue, mean_green, median_green, std_green, brightness_green, mean_red, median_red, std_red, brightness_red = extract_color_features(color_img)  # Reading in color for feature extraction
    vessels_image = extract_blood_vessels(img)
    # Invert the mask to focus on the region of interest
    mask_ = cv2.bitwise_not(background_mask)
    # Reduce the radius of the mask by 10 pixels using erosion
    kernel_reduce = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))  # Kernel size slightly larger than 10 pixels
    mask_reduced = cv2.erode(mask_, kernel_reduce, iterations=1)
    # Apply the mask to the image
    vessels_image = cv2.bitwise_and(vessels_image, img, mask=mask_reduced)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # vessels_image = cv2.morphologyEx(vessels_image, cv2.MORPH_OPEN, kernel)
    vessel_area_ratio = calculate_vessel_area(vessels_image)
    contrast, correlation, energy, homogeneity, entropy = extract_texture_features(img)
    edge_density = extract_shape_features(color_img)
    mean_gradient, std_gradient = extract_gradient_features(color_img)
    skewness, kurt = extract_histogram_features(color_img)
    local_binary_patterns = extract_lbp_features(img)

    urls = save_images_and_get_urls([img, background_mask, mask, vessels_image, mask_])

    return np.array([central_opacity, overall_opacity, vessel_area_ratio,
                     mean_blue, median_blue, std_blue, brightness_blue, mean_green,
                     median_green, std_green, brightness_green, mean_red, median_red, std_red, brightness_red,
                     contrast, correlation, energy, homogeneity, entropy,
                     edge_density,
                     mean_gradient, std_gradient,
                     skewness, kurt,
                     *local_binary_patterns
                     ]).flatten(), urls