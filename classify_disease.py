from classify_glaucoma import predict_glaucoma
from classify_diabetes_retinopathy import predict_diabetes_retinopathy
from classify_cataract import predict_cataract
from classify_retinal_detachment import predict_retinal_detachment
from extract_other_fundus import extract_common_features
import cv2
import numpy as np
from skimage import color
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern

def resize_image(image, size=(512, 512)):
    return cv2.resize(image, size)

def calculate_features(candidate_segment):
    # Convert candidate segment to float
    candidate_segment = candidate_segment.astype(float)
    # Energy
    energy = np.sum(candidate_segment ** 2)
    # Gradient calculation
    gradient_x = cv2.Sobel(candidate_segment, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(candidate_segment, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    # Mean Gradient
    mean_gradient = np.mean(gradient_magnitude)
    # Standard Deviation Gradient
    std_gradient = np.std(gradient_magnitude)
    # Mean Intensity
    mean_intensity = np.mean(candidate_segment)
    # Intensity Variation
    std_intensity = np.std(candidate_segment)
    intensity_variation = mean_intensity / (std_intensity + 1e-10)  # Adding a small epsilon to avoid division by zero
    return np.array([energy, mean_gradient, std_gradient, mean_intensity, intensity_variation])

def compute_texture_features(image):
    gray_image = color.rgb2gray(image)
    # Compute the GLCM (Gray-Level Co-occurrence Matrix)
    glcm = graycomatrix((gray_image * 255).astype(np.uint8), [1], [0], symmetric=True, normed=True)
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    entropy = -np.sum(energy * np.log2(energy + 1e-10))
    return np.array([homogeneity, contrast, energy, entropy])

def compute_entropy_g(image):
    b, g, r = cv2.split(image)
    # Compute the histogram
    histogram, _ = np.histogram(g.flatten(), bins=256, range=[0,256])
    # Normalize the histogram to get the probability distribution
    probability_distribution = histogram / histogram.sum()
    # Compute the entropy
    entropy_g = -np.sum(probability_distribution * np.log2(probability_distribution + 1e-10))
    return np.array([entropy_g])

def std_rgb(image):
    b, g, r = cv2.split(image)
    std_r = np.std(r)
    std_g = np.std(g)
    std_b = np.std(b)
    return np.array([std_r, std_g, std_b])

def rgb_to_magenta(image):
    cmy_image = 1 - image / 255.0
    magenta_image = cmy_image[:, :, 1]
    return magenta_image

def morphological_vessels(image):
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=1)
    opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
    dilated = cv2.dilate(opened, kernel, iterations=1)
    return dilated

def subtract_images(original, morphed):
    return cv2.subtract(original, morphed)

def histogram_equalization(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(image)
    return equalized

def binarize_image(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    _, binary_image = cv2.threshold(image, 17, 255, cv2.THRESH_BINARY)
    return binary_image

def noise_reduct(binary_image):
    kernel = np.ones((1, 1), np.uint8)
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    num_labels, labels_im = cv2.connectedComponents(closed)
    for i in range(1, num_labels):
        if np.sum(labels_im == i) < 100:
            closed[labels_im == i] = 0
    return closed

def skeletonize(image):
    return cv2.ximgproc.thinning(image)

def analyze_tortuosity_lenght(skeleton):
    # Identify individual vessel segments
    num_labels, labels_im = cv2.connectedComponents(skeleton)
    abnormal_vessels = np.zeros_like(skeleton)
    # Analyze each vessel segment
    for i in range(1, num_labels):
        segment = np.zeros_like(skeleton)
        segment[labels_im == i] = 255
        # Measure the length of the segment
        length = np.sum(segment > 0)
        # Measure the tortuosity (sum of angles between successive pixels)
        coords = np.column_stack(np.where(segment > 0))
        if len(coords) < 5:
            continue
        total_angle = 0
        for j in range(1, len(coords) - 1):
            p1 = coords[j - 1]
            p2 = coords[j]
            p3 = coords[j + 1]
            v1 = p2 - p1
            v2 = p3 - p2
            angle = np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
            total_angle += np.abs(angle)
        tortuosity = total_angle / length
        # High tortuosity and shorter length indicates new abnormal vessels
        if tortuosity > 0.7 and length < 120:
            abnormal_vessels[labels_im == i] = 255
    return abnormal_vessels

def compute_density(image):
    white_pixels = np.sum(image == 255)
    total_pixels = image.size
    density = white_pixels / total_pixels
    return density

def detect_neovascularization(image):
    magenta_component = rgb_to_magenta(image)
    morphed_image = morphological_vessels(magenta_component)
    subtracted_image = subtract_images(magenta_component, morphed_image)
    enhanced_image = histogram_equalization(subtracted_image)
    binary_image = binarize_image(enhanced_image)
    vessels = noise_reduct(binary_image)
    skeleton = skeletonize(vessels)
    abnormal_vessels = analyze_tortuosity_lenght(skeleton)
    abnormal_vessel_density = compute_density(abnormal_vessels)
    return np.array([abnormal_vessel_density])

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

def apply_fundus_mask(image, margin):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  _, thresholded = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
  contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # Sort contours by area and find the largest contour (assumed to be the optic disc)
  largest_contour = max(contours, key=cv2.contourArea)
  initial_fundus_mask = np.zeros_like(image)
  cv2.drawContours(initial_fundus_mask, [largest_contour], -1, (255, 255, 255), cv2.FILLED)
  height, width = initial_fundus_mask.shape[:2]
  # Define circle parameters (center and radius)
  center = (width // 2, height // 2)
  radius = max(center)-margin
  # Create circular mask
  fundus_mask = np.zeros((height, width), dtype=np.uint8)
  cv2.circle(fundus_mask, center, radius, 255, thickness=cv2.FILLED)
  masked_image = np.zeros_like(image)
  masked_image[fundus_mask == 255] = image[fundus_mask == 255]
  return masked_image, fundus_mask

def localize_disc(image):
    image = resize_image(image)
    img = image.copy()
    image, _ = apply_fundus_mask(image, 400)
    green_channel = image[:, :, 1]
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened_green_channel = cv2.morphologyEx(green_channel, cv2.MORPH_OPEN, kernal)
    clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(8,8))
    clahe_green_channel = clahe.apply(opened_green_channel)
    max_value = np.max(clahe_green_channel)
    _, thresholded = cv2.threshold(clahe_green_channel, max_value - 10, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area and find the largest contour (assumed to be the optic disc)
    largest_contour = max(contours, key=cv2.contourArea)
    # Find the bounding box of the optic disc
    x, y, w, h = cv2.boundingRect(largest_contour)
    margin = 300
    # Extract the region of interest (ROI) containing the optic disc
    optic_disc_roi = img[y-margin:y+h+margin, x-margin:x+w+margin]
    return optic_disc_roi

def segment_fluid_areas_rrd(image, mask):
    # Increase brightness
    brightness_increase = 20
    brightened = cv2.add(image, np.ones(image.shape, dtype=np.uint8) * brightness_increase)
    # Convert to LAB color space
    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
     # Convert to RGB color space and split channels
    rgb = cv2.cvtColor(brightened, cv2.COLOR_BGR2RGB)
    r, g, b_rgb = cv2.split(rgb)
    # Threshold the 'L' channel
    _, dark_spots = cv2.threshold(r, 105, 255, cv2.THRESH_BINARY_INV)
    # Apply the mask to the thresholded image
    dark_spots = cv2.bitwise_and(dark_spots, dark_spots, mask=mask)
    # Clean up the thresholded image using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(dark_spots, cv2.MORPH_OPEN, kernel, iterations=2)
    return brightened, cleaned

def segment_dark_spots_with_texture_rrd(image, mask):
    # Apply LAB-based segmentation
    brightened, lab_segmented = segment_fluid_areas_rrd(image, mask)
    # Convert to grayscale
    gray = cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY)
    # Apply LBP for texture segmentation
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    # Threshold the LBP image to segment dark textures
    _, lbp_segmented = cv2.threshold(lbp, np.percentile(lbp, 95), 255, cv2.THRESH_BINARY_INV)
    # Combine the LAB-based and LBP-based segmentations
    combined_segmented = cv2.bitwise_and(lab_segmented, lbp_segmented.astype(np.uint8))
    # Apply the mask to the combined segmented image
    combined_segmented = cv2.bitwise_and(combined_segmented, combined_segmented, mask=mask)
    # Clean up the combined segmented image using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_combined = cv2.morphologyEx(combined_segmented, cv2.MORPH_OPEN, kernel, iterations=2)
    return brightened, cleaned_combined

def cluster_image(image, k):
    image = cv2.medianBlur(image, 7)
    # Reshape the image to a 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    # Define criteria and apply KMeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Convert centers back to 8-bit values
    centers = np.uint8(centers)
    # Map the labels to the center values
    segmented_image = centers[labels.flatten()]
    # Reshape back to the original image dimensions
    clustered_image = segmented_image.reshape(image.shape)
    return clustered_image, labels, centers

def segment_optic_cup(image):
    # Step 1: Cluster the image into 3 clusters
    _, labels, centers = cluster_image(image, 3)
    # Step 2: Identify the brightest cluster
    brightness = np.sum(centers, axis=1)  # Sum the RGB values to determine brightness
    brightest_cluster_index = np.argmax(brightness)  # Find the index of the brightest cluster
    # Step 3: Create a mask for the brightest cluster
    brightest_cluster_mask = (labels == brightest_cluster_index).astype(np.uint8)
    # Reshape the mask to match the original image dimensions
    brightest_cluster_mask = brightest_cluster_mask.reshape(image.shape[:2])
    # Step 4: Return the binary mask of the brightest cluster
    binary_mask = (brightest_cluster_mask * 255).astype(np.uint8)
    return 

def cluster_optic_disc(image):
    image = image[:,:,2]
    image = cv2.equalizeHist(image)
    image = cv2.medianBlur(image, 7)
    pixel_values = image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 2
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    clustered_image = segmented_image.reshape(image.shape)
    _, binary_image = cv2.threshold(clustered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def calculate_overall_cdr(cup_image, disc_image):
    cup_area = cv2.countNonZero(cup_image)
    disc_area = cv2.countNonZero(disc_image)
    cdr = cup_area / disc_area
    return str(cdr)

def calculate_vessel_density_rd(segment_mask):
    total_pixels = segment_mask.size
    vessel_pixels = np.sum(segment_mask == 255)
    vessel_density = (vessel_pixels / total_pixels) * 100
    return vessel_density

def predict_disease(img, file):
    if file is not None:
        neovascularization_features = detect_neovascularization(img)
        mask = create_central_mask(img)
        central_opacity = measure_opacity(img, mask)
        overall_opacity = measure_opacity(img)
        optic_disc_roi = localize_disc(img)
        disc = cluster_optic_disc(optic_disc_roi)
        cup = segment_optic_cup(optic_disc_roi)
        cdr = calculate_overall_cdr(disc, cup)
        blood_vessel_density = calculate_vessel_density_rd(img, mask) 
        predict = None
        if cdr > 0.6:
            predict = predict_glaucoma
        elif neovascularization_features[0] > 0:
            predict = predict_diabetes_retinopathy
        elif central_opacity < 80 or overall_opacity < 80:
            predict = predict_cataract
        elif blood_vessel_density < 0.01:
            predict = predict_retinal_detachment
        else:
            predict = extract_common_features
        return predict(img)
    else:
        return Exception("File is not valid.")