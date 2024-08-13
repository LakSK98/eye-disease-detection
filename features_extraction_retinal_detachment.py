import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from save_processed_images import save_images_and_get_urls

def segment_retinal_area_rd(image, color='white'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    if color == 'black':
        background_color = (0, 0, 0)
    elif color == 'white':
        background_color = (255, 255, 255)
    else:
        raise ValueError("Color parameter error")
    segmented_image = np.full_like(image, background_color)
    segmented_image[mask == 255] = image[mask == 255]
    return segmented_image, mask

def segment_blood_vessels_rd(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    blue, green, red = cv2.split(final)
    r1 = cv2.morphologyEx(green, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    f4 = cv2.subtract(R3, green)
    f5 = clahe.apply(f4)
    image1 = f5
    e_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closeImg = cv2.morphologyEx(image1, cv2.MORPH_CLOSE, e_kernel)
    revImg = closeImg
    topHat = image1 - revImg
    imge = topHat
    blur = cv2.GaussianBlur(imge, (5, 5), 0)
    ret, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, f6 = cv2.threshold(f5, 15, 255, cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype='uint8') * 255
    contours, hierarchy = cv2.findContours(f6.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 255:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret, fin = cv2.threshold(im, 15, 255, cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    fundus_eroded = cv2.bitwise_not(newfin)
    xmask = np.ones(fundus_eroded.shape[:2], dtype='uint8') * 255
    xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in xcontours:
        shape = 'unidentified'
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = 'circle'
        else:
            shape = 'vessels'
        if shape == 'circle':
            cv2.drawContours(xmask, [cnt], -1, 0, -1)
    finimage = cv2.bitwise_and(fundus_eroded, fundus_eroded, mask=xmask)
    blood_vessels = cv2.bitwise_not(finimage)
    kernel = np.ones((2, 2), np.uint8)
    blood_vessels = cv2.subtract(255, blood_vessels)
    new = cv2.morphologyEx(blood_vessels, cv2.MORPH_OPEN, kernel)
    new1 = cv2.morphologyEx(new, cv2.MORPH_CLOSE, kernel)
    return new1

def remove_vessels_rd(img, vessel_mask):
    inpainted_img = cv2.inpaint(img, vessel_mask, 20, cv2.INPAINT_TELEA)
    smoothed_inpainted_img = cv2.GaussianBlur(inpainted_img, (15, 15), 0)
    return smoothed_inpainted_img

def divide_into_segments_rd(mask, center, num_segments=12):
    height, width = mask.shape
    radius = min(center[0], center[1], width - center[0], height - center[1])
    segments = []
    angle = 360 / num_segments
    for i in range(num_segments):
        segment_mask = np.zeros_like(mask)
        start_angle = i * angle
        end_angle = (i + 1) * angle
        cv2.ellipse(segment_mask, center, (radius, radius), 0, start_angle, end_angle, 255, -1)
        segments.append(segment_mask)
    return segments

def calculate_vessel_density_rd(segment_mask):
    total_pixels = segment_mask.size
    vessel_pixels = np.sum(segment_mask == 255)
    vessel_density = (vessel_pixels / total_pixels) * 100
    return vessel_density

def calculate_center_rd(mask):
    moments = cv2.moments(mask, binaryImage=True)
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])
    center = (cX, cY)
    return center

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

def calculate_fluid_area_percentage_rrd(segmented, mask):
    # Calculate the retinal area (non-black area)
    retinal_area = np.sum(mask >= 0)
    # Calculate the fluid area
    fluid_area = np.sum(segmented > 0)
    # Calculate the percentage of the fluid area
    fluid_area_percentage = (fluid_area / retinal_area) * 100
    return fluid_area_percentage

def preprocess_image_erd(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

def segment_exudates_erd(image):
    #Segment exudates in the preprocessed image using thresholding and morphological operations.
    preprocessed = preprocess_image_erd(image)
    _, thresh = cv2.threshold(preprocessed, 184.7, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    exudate_pixels = (np.sum(morph)/255)
    total_pixels = (np.sum(preprocessed)/255)
    exudate_ratio = exudate_pixels / total_pixels
    return exudate_pixels,total_pixels,exudate_ratio,morph

def main_erd(image):
    exudate_pixels,total_pixels,exudate_ratio, exudate_mask = segment_exudates_erd(image)
    return exudate_pixels,exudate_ratio,exudate_mask

def calculate_color_percentage_trd(image):
    color_ranges = [
    ((18, 100, 60), (30, 255, 255)),  #8A6526
    ((18, 100, 100), (30, 255, 255)),  #8E803E
    ((18, 150, 80), (30, 255, 255)),  #97661A
    ((18, 100, 100), (30, 255, 255)),  #A7854D
    ((25, 100, 180), (35, 255, 255)),  #D8B741
    ((25, 100, 180), (35, 255, 255)),  #D9AE23
    ((18, 150, 80), (30, 255, 255)),  #A27425
    ((18, 100, 100), (30, 255, 255)),  #9D8431
    ((18, 150, 80), (30, 255, 255)),  #997220
    ((18, 100, 100), (30, 255, 255)),  #AF9347
    ((18, 150, 80), (30, 255, 255)),  #946A1C
    ((25, 100, 180), (35, 255, 255)),  #B69736
    ((25, 100, 180), (35, 255, 255)),  #B68C2F
    ((18, 100, 100), (30, 255, 255))   #A88634
]
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for (lower, upper) in color_ranges:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        temp_mask = cv2.inRange(hsv_image, lower, upper)
        mask = cv2.bitwise_or(mask, temp_mask)
    color_pixel_count = np.sum(mask == 255)
    total_pixel_count = image.shape[0] * image.shape[1]
    color_percentage = (color_pixel_count / total_pixel_count) * 100
    # Highlight the color pixels in white for visualization
    highlighted_image = np.zeros_like(image)
    highlighted_image[mask == 255] = [255, 255, 255]
    return color_percentage, highlighted_image

def extract_features(img):
    segmented_image, mask = segment_retinal_area_rd(img)
    vessel_mask = segment_blood_vessels_rd(segmented_image)
    smoothed_inpainted_img = remove_vessels_rd(segmented_image, vessel_mask)
    center = calculate_center_rd(mask)
    segments = divide_into_segments_rd(vessel_mask, center)
    rd_feature = []
    density_list = []
    vessels_segments = []
    for i, segment in enumerate(segments):
        segmented_image = cv2.bitwise_and(vessel_mask, vessel_mask, mask=segment)
        vessels_segments.append(segmented_image)
        density = calculate_vessel_density_rd(segmented_image)
        density_list.append(density)
        rd_feature.append(density)
    total_density = sum(density_list)
    average_density = total_density / len(density_list)
    rd_feature.append(total_density)
    rd_feature.append(average_density)
    brightened, segmented_dark_spots = segment_dark_spots_with_texture_rrd(smoothed_inpainted_img, mask)
    contours, _ = cv2.findContours(segmented_dark_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fluid_areas_count = len(contours)
    rd_feature.append(fluid_areas_count)
    fluid_area_percentage = calculate_fluid_area_percentage_rrd(segmented_dark_spots, mask)
    rd_feature.append(fluid_area_percentage)
    exudate_pixels, exudate_ratio, exudate_mask = main_erd(img)
    rd_feature.append(exudate_ratio)
    color_percentage, highlighted_image = calculate_color_percentage_trd(img)
    rd_feature.append(color_percentage)
    
    urls = save_images_and_get_urls([segmented_image, mask, vessel_mask, smoothed_inpainted_img, segmented_dark_spots, exudate_mask, highlighted_image, brightened] + vessels_segments)

    return np.array(rd_feature), urls