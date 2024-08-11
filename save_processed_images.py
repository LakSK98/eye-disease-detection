import os
import uuid
from flask import url_for
import numpy as np

# Define the directory to save images
UPLOAD_FOLDER = 'static/processed/'

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def save_image(image, prefix='image'):
    # Generate a unique filename using uuid
    filename = f"{prefix}_{uuid.uuid4().hex}.png"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    # Save the image (assuming image is a PIL Image object or a NumPy array)
    if isinstance(image, np.ndarray):
        from PIL import Image
        image = Image.fromarray(image)  
    image.save(filepath)
    # Generate the URL for the saved image
    url = url_for('static', filename=f'processed/{filename}', _external=True)
    return url

def save_images_and_get_urls(images_array):
    urls = []
    # Iterate through the array of images and save each one
    for idx, image in enumerate(images_array):
        prefix = f'image_{idx}'
        url = save_image(image, prefix)
        urls.append(url)
    return urls