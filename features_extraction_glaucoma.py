import cv2
import numpy as np
import scipy.stats
from save_processed_images import save_images_and_get_urls

def resize_image(image):
  original_height, original_width = image.shape[:2]
  # Conditionally upscale the image if width or height is below 1000 pixels
  if original_width < 1000 or original_height < 1000:
    upscale_factor = 3
    upscaled_width = original_width * upscale_factor
    upscaled_height = original_height * upscale_factor
    image = cv2.resize(image, (upscaled_width, upscaled_height), interpolation=cv2.INTER_CUBIC)
  return image

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

def segment_adaptive_optic_disc(input_image):
    red_channel = input_image[:, :, 2]
    equalized_red_channel = cv2.equalizeHist(red_channel)
    max_intensity = np.max(red_channel)
    total_pixels = red_channel.size
    histogram = cv2.calcHist([equalized_red_channel], [0], None, [256], [0, 256]).flatten()
    threshold_value = max_intensity-1
    while threshold_value > 0:
        count_pixels = np.sum(histogram[threshold_value:])
        if count_pixels / total_pixels >= 0.28:
            break
        threshold_value -= 1
    _, binary_image = cv2.threshold(equalized_red_channel, threshold_value, 255, cv2.THRESH_BINARY)
    _, otsu_threshold = cv2.threshold(red_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image = cv2.bitwise_and(binary_image, otsu_threshold)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    return binary_image

def remove_small_objects(binary_image, min_size):
    # Perform connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    # Create an output image initialized to zero (black)
    output_image = np.zeros(binary_image.shape, dtype=np.uint8)
    # Loop through all components and remove small objects
    for i in range(1, num_labels):  # Start from 1 to skip the background
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output_image[labels == i] = 255
    return output_image

def extract_blood_vessels(image):
  # Extract the green channel
  green_channel = image[:, :, 1]
  structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
  black_hat = cv2.morphologyEx(green_channel, cv2.MORPH_BLACKHAT, structuring_element)
  _, vessel_image = cv2.threshold(black_hat, 8, 255, cv2.THRESH_BINARY)
  vessel_image = remove_small_objects(vessel_image, 550)
  kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  closed_vessels = cv2.morphologyEx(vessel_image, cv2.MORPH_CLOSE, kernal)
  return vessel_image, closed_vessels

def remove_vessels(img):
  vessels_mask, smoothed_vessels_mask = extract_blood_vessels(img)
  removed_vessels = np.zeros_like(img)
  removed_vessels[smoothed_vessels_mask == 0] = img[smoothed_vessels_mask == 0]
  vessel_inpainted_img = cv2.inpaint(removed_vessels, smoothed_vessels_mask, 10, cv2.INPAINT_TELEA)
  removed_vessel_image = cv2.GaussianBlur(vessel_inpainted_img, (11, 11), 0)
  return removed_vessel_image

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

def select_centered_contour(image):
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (151, 151)))
    # Convert the image to grayscale if it's not already
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Threshold the image to create a binary image
    _, binary_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get the center point of the image
    height, width = binary_img.shape
    center_point = (width // 2, height // 2)
    # Initialize a list to store contours that contain the center point
    selected_contours = []
    # Iterate over contours and check if they contain the center point
    for contour in contours:
        if cv2.pointPolygonTest(contour, center_point, False) >= 0:
            selected_contours.append(contour)
    # Create an empty mask to draw the selected contours
    output_image = np.zeros_like(image)
    ellipse = cv2.fitEllipse(selected_contours[0])
    cv2.ellipse(output_image, ellipse, (255, 255, 255), thickness=cv2.FILLED)
    return output_image

def segment_optic_disc(inpainted_roi):
  disc_img = segment_adaptive_optic_disc(inpainted_roi)
  clut_bin_img = cluster_optic_disc(inpainted_roi)
  disc = cv2.bitwise_and(clut_bin_img, disc_img)
  disc = select_centered_contour(disc)
  return disc, disc_img, clut_bin_img

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
    return binary_mask

def neuroretinal_rim(optic_disc, optic_cup):
  neuroretinal_rim = cv2.bitwise_and(optic_disc, cv2.bitwise_not(optic_cup))
  neuroretinal_rim = crop_image(neuroretinal_rim)
  return neuroretinal_rim

def istn_masks(neuroretinal_rim_gray):
  height, width = neuroretinal_rim_gray.shape
  center_x, center_y = width // 2, height // 2
  # Calculate the angle of each pixel relative to the center of the image
  y, x = np.indices((height, width))
  angle = np.arctan2(y - center_y, x - center_x) * 180 / np.pi
  angle[angle < 0] += 360
  # Create masks for the four quadrants
  inferior_mask = (((angle >= 45) & (angle < 135)) * 255).astype(np.uint8)
  superior_mask = (((angle >= 225) & (angle < 315)) * 255).astype(np.uint8)
  nasal_mask =  (((angle >= 135) & (angle < 225)) * 255).astype(np.uint8)
  temporal_mask =  (((angle >= 315) | (angle < 45)) * 255).astype(np.uint8)
  return inferior_mask, superior_mask, nasal_mask, temporal_mask

def istn_rule(neuroretinal_rim_gray):
  inferior_mask, superior_mask, nasal_mask, temporal_mask = istn_masks(neuroretinal_rim_gray)
  # Combine masks to get the specific quadrants
  inferior = cv2.bitwise_and(inferior_mask.astype(np.uint8), neuroretinal_rim_gray)
  superior = cv2.bitwise_and(superior_mask.astype(np.uint8), neuroretinal_rim_gray)
  nasal = cv2.bitwise_and(nasal_mask.astype(np.uint8), neuroretinal_rim_gray)
  temporal = cv2.bitwise_and(temporal_mask.astype(np.uint8), neuroretinal_rim_gray)
  # Calculate the thickness (area) in each quadrant
  inferior_thickness = np.sum(inferior == 255)
  superior_thickness = np.sum(superior == 255)
  nasal_thickness = np.sum(nasal == 255)
  temporal_thickness = np.sum(temporal == 255)
  # Implement the ISNT rule
  if inferior_thickness > superior_thickness > nasal_thickness > temporal_thickness:
      return 1, inferior_thickness , superior_thickness , nasal_thickness , temporal_thickness, inferior , superior , nasal , temporal
  else:
      return 0, inferior_thickness , superior_thickness , nasal_thickness , temporal_thickness, inferior , superior , nasal , temporal

def segment_ppa(roi, disc_mask):
    contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No disc contours found in the disc mask.")
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    margin = 50
    x_start = max(x - margin, 0)
    y_start = max(y - margin, 0)
    x_end = min(x + w + margin, disc_mask.shape[1])
    y_end = min(y + h + margin, disc_mask.shape[0])
    cropped_disc_mask = disc_mask[y_start:y_end, x_start:x_end]
    cropped_roi = roi[y_start:y_end, x_start:x_end, :]
    disc_removed = np.zeros_like(cropped_roi)
    disc_removed[cropped_disc_mask == 0] = cropped_roi[cropped_disc_mask == 0]
    ppa_mask = disc_removed[:, :, 2]
    equalized_ppa_mask = cv2.equalizeHist(ppa_mask)
    _, finalized_ppa = cv2.threshold(equalized_ppa_mask, 220, 255, cv2.THRESH_BINARY)
    return finalized_ppa

def crop_image(image):
  # Find contours of the shape
  contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # Assume the largest contour is the shape to be cropped
  x, y, w, h = cv2.boundingRect(contours[0])
  # Crop the image to the bounding box of the shape
  cropped_img = image[y:y+h, x:x+w]
  return cropped_img

def blood_vessels_density(img):
  blood_vessels_mask, closed_blood_vessels_mask = extract_blood_vessels(img)
  vessel_density = cv2.countNonZero(closed_blood_vessels_mask) / closed_blood_vessels_mask.size
  return vessel_density, blood_vessels_mask

def flip_image(image):
    if image is None or len(image.shape) == 0:
        raise ValueError("Invalid input image")
    # Perform horizontal flip
    horizontal_flip = cv2.flip(image, 1)
    # Perform vertical flip
    vertical_flip = cv2.flip(image, 0)
    return vertical_flip, horizontal_flip

def disc_assymmetry(disc_mask):
    vertical_flip, horizontal_flip = flip_image(disc_mask)
    subtract_vertical = cv2.bitwise_and(disc_mask, cv2.bitwise_not(vertical_flip))
    subtract_horizontal = cv2.bitwise_and(disc_mask, cv2.bitwise_not(horizontal_flip))
    subtract_vertical_density = cv2.countNonZero(subtract_vertical) / disc_mask.size
    subtract_horizontal_density = cv2.countNonZero(subtract_horizontal) / disc_mask.size
    return subtract_vertical_density, subtract_horizontal_density

def blood_vessels_with_color(original_img, vessels_mask):
  segmented_color_vessels = np.zeros_like(original_img)
  segmented_color_vessels[vessels_mask == 255] = original_img[vessels_mask == 255]
  return segmented_color_vessels

def calculate_overall_cdr(cup_image, disc_image):
    cup_area = cv2.countNonZero(cup_image)
    disc_area = cv2.countNonZero(disc_image)
    cdr = cup_area / disc_area
    return str(cdr)

def calculate_vertical_cdr(cup_image, disc_image):
    cup_height = cv2.countNonZero(cv2.reduce(cup_image, 1, cv2.REDUCE_MAX))
    disc_height = cv2.countNonZero(cv2.reduce(disc_image, 1, cv2.REDUCE_MAX))
    vertical_cdr = cup_height / disc_height
    return str(vertical_cdr)

def calculate_horizontal_cdr(cup_image, disc_image):
    cup_width = cv2.countNonZero(cv2.reduce(cup_image, 0, cv2.REDUCE_MAX))
    disc_width = cv2.countNonZero(cv2.reduce(disc_image, 0, cv2.REDUCE_MAX))
    horizontal_cdr = cup_width / disc_width
    return str(horizontal_cdr)

def calculate_cdr(cup_image, disc_image):
    cdr = calculate_overall_cdr(cup_image, disc_image)
    vertical_cdr = calculate_vertical_cdr(cup_image, disc_image)
    horizontal_cdr = calculate_horizontal_cdr(cup_image, disc_image)
    return cdr, vertical_cdr, horizontal_cdr

def calculate_neuroretinal_rim_thickness(disc_mask, cup_mask):
    # Find contours of the optic disc
    disc_contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(disc_contours) == 0:
        return None  # No contour found
    # Assume the largest contour is the optic disc
    disc_contour = max(disc_contours, key=cv2.contourArea)
    # Find contours of the optic cup
    cup_contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cup_contours) == 0:
        return None  # No contour found
    # Assume the largest contour is the optic cup
    cup_contour = max(cup_contours, key=cv2.contourArea)
    # Calculate the minimum distance from each point on the optic cup to the optic disc
    distances = []
    for point in cup_contour:
        point = tuple(point[0])  # Ensure point is a tuple (x, y)
        distance = cv2.pointPolygonTest(disc_contour, (float(point[0]), float(point[1])), True)
        if distance > 0:  # Positive distances are inside the disc contour
            distances.append(distance)
    # Calculate the average thickness
    if len(distances) > 0:
        average_thickness = np.mean(distances)
    else:
        average_thickness = 0  # In case no valid distances were found
    return average_thickness

def red_color_features_blood_vessels(img, blood_vessels_mask):
    color_blood_vessels = blood_vessels_with_color(img, blood_vessels_mask)
    # Convert the image to different color spaces
    hsv_image = cv2.cvtColor(color_blood_vessels, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(color_blood_vessels, cv2.COLOR_BGR2Lab)
    # BGR space: Focus on the Red channel
    mean_red = np.mean(color_blood_vessels[:,:,2])  # Red channel is the third channel in BGR
    std_red = np.std(color_blood_vessels[:,:,2])
    max_red = np.max(color_blood_vessels[:,:,2])
    min_red = np.min(color_blood_vessels[:,:,2])
    # Calculate color skewness and kurtosis in the Red channel
    skew_red = scipy.stats.skew(color_blood_vessels[:,:,2].flatten())
    kurt_red = scipy.stats.kurtosis(color_blood_vessels[:,:,2].flatten())
    # HSV space: Focus on the Hue channel (which represents color)
    hue_channel = hsv_image[:,:,0]
    mean_hue = np.mean(hue_channel)
    std_hue = np.std(hue_channel)
    # Lab space: Focus on the 'a' channel (which represents red-green)
    mean_a = np.mean(lab_image[:,:,1])
    std_a = np.std(lab_image[:,:,1])
    return mean_red, std_red, max_red, min_red, skew_red, kurt_red, mean_hue, std_hue, mean_a, std_a

def red_color_features_disc(disc_with_color):
    # Convert the image to different color spaces
    hsv_image = cv2.cvtColor(disc_with_color, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(disc_with_color, cv2.COLOR_BGR2Lab)
    # BGR space: Focus on the Red channel
    mean_red = np.mean(disc_with_color[:,:,2])  # Red channel is the third channel in BGR
    std_red = np.std(disc_with_color[:,:,2])
    max_red = np.max(disc_with_color[:,:,2])
    min_red = np.min(disc_with_color[:,:,2])
    # Calculate color skewness and kurtosis in the Red channel
    skew_red = scipy.stats.skew(disc_with_color[:,:,2].flatten())
    kurt_red = scipy.stats.kurtosis(disc_with_color[:,:,2].flatten())
    # HSV space: Focus on the Hue channel (which represents color)
    hue_channel = hsv_image[:,:,0]
    mean_hue = np.mean(hue_channel)
    std_hue = np.std(hue_channel)
    # Lab space: Focus on the 'a' channel (which represents red-green)
    mean_a = np.mean(lab_image[:,:,1])
    std_a = np.std(lab_image[:,:,1])
    return mean_red, std_red, max_red, min_red, skew_red, kurt_red, mean_hue, std_hue, mean_a, std_a

def density_of_ppa(roi, disc_mask):
    ppa_mask = segment_ppa(roi, disc_mask)
    ppa_density = cv2.countNonZero(ppa_mask) / ppa_mask.size
    return ppa_density, ppa_mask

def segment_disc_with_color(roi, disc_mask):
    disc_with_color = np.zeros_like(roi)
    disc_with_color[disc_mask == 255] = roi[disc_mask == 255]
    return disc_with_color

def extract_features(img):
    optic_disc_roi = localize_disc(img)
    inpainted_roi = remove_vessels(optic_disc_roi)
    disc, d, cl = segment_optic_disc(inpainted_roi)
    inpainted_disc_seg = segment_disc_with_color(inpainted_roi, disc)
    cup = segment_optic_cup(inpainted_disc_seg)
    cdr, vertical_cdr, horizontal_cdr = calculate_cdr(cup, disc)
    neuroretinal_rim_mask = neuroretinal_rim(disc, cup)
    is_istn_rule_follows, inferior_thickness , superior_thickness , nasal_thickness , temporal_thickness, inferior , superior , nasal , temporal = istn_rule(neuroretinal_rim_mask)
    average_neuroretinal_rim_thickness = calculate_neuroretinal_rim_thickness(disc, cup)
    disc_assym_vertical, disc_assym_horizontal = disc_assymmetry(disc)
    vessels_density, blood_vessels_mask = blood_vessels_density(img)
    mean_red, std_red, max_red, min_red, skew_red, kurt_red, mean_hue, std_hue, mean_a, std_a = red_color_features_blood_vessels(img, blood_vessels_mask)
    ppa_density, ppa_mask = density_of_ppa(inpainted_roi, disc)
    disc_seg = segment_disc_with_color(optic_disc_roi, disc)
    mean_red_disc, std_red_disc, max_red_disc, min_red_disc, skew_red_disc, kurt_red_disc, mean_hue_disc, std_hue_disc, mean_a_disc, std_a_disc = red_color_features_disc(disc_seg)
    processed_urls = save_images_and_get_urls([optic_disc_roi, disc, cup, inpainted_disc_seg, neuroretinal_rim_mask, blood_vessels_mask, inpainted_roi, ppa_mask,
                                               inferior , superior , nasal , temporal])
    return np.array([cdr, vertical_cdr, horizontal_cdr,
            is_istn_rule_follows, inferior_thickness , superior_thickness , nasal_thickness , temporal_thickness,
            average_neuroretinal_rim_thickness,
            disc_assym_vertical, disc_assym_horizontal,
            vessels_density,
            mean_red, std_red, max_red, min_red, skew_red, kurt_red, mean_hue, std_hue, mean_a, std_a,
            ppa_density,
            mean_red_disc, std_red_disc, max_red_disc, min_red_disc, skew_red_disc, kurt_red_disc, mean_hue_disc, std_hue_disc, mean_a_disc, std_a_disc]).flatten(), processed_urls
