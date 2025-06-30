import os
import numpy as np
import cv2
from glob import glob
from skimage.measure import label, regionprops
from skimage.transform import hough_circle, hough_circle_peaks
import cv2


# -------------------------- Essential Functions ---------------------------

def crop_from_top_to_circle(image, top_of_circle, margin=0.1):
    """
    Crops the image from the top up to slightly above the detected circle.

    Parameters:
      image: Input image (e.g., mask, as a 2D numpy array).
      top_of_circle: The y-coordinate of the highest point of the circle.
      margin: Fraction of top_of_circle to add as extra margin.

    Returns:
      Cropped mask.
    """
    height, _ = image.shape[:2]
    # Define the top crop: a little above the circle's top
    top_crop = max(0, top_of_circle - int(margin * top_of_circle))
    # Crop from the top to that point
    return image[0:top_crop, :]


def crop_from_top_to_circle_rgb(image, top_of_circle, margin=0.1):
    """
    Crop from the top of the image up to just above the circle.

    Parameters:
      image: Input RGB image (3D numpy array with shape (H, W, 3)).
      top_of_circle: The y-coordinate of the highest point of the circle.
      margin: Fraction of top_of_circle to add as extra margin.

    Returns:
      Cropped RGB image.
    """
    height, width, channels = image.shape
    top_crop = max(0, top_of_circle - int(margin * top_of_circle))
    return image[0:top_crop, :, :]


def roi_crop(image_file, mask_file):
    """
    Crops the circle region of the image and its corresponding mask.

    Parameters:
      image_file: RGB image (BGR as read by cv2) to be cropped.
      mask_file: Mask image (must have the same dimensions as image_file).

    Returns:
      cropped_img: Cropped RGB image.
      cropped_mask: Cropped mask.
    """

    # Convert image to grayscale
    gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    # Apply a small Gaussian blur
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    # Edge detection
    edges = cv2.Canny(gray_blurred, 10, 10)

    # Ensure the mask and image have the same dimensions
    if mask_file.shape != image_file.shape:
        raise ValueError(
            f"Dimensions do not match in Mask: {os.path.basename(str(mask_file))} and Image: {os.path.basename(str(image_file))}")

    # Detect circles using the Hough transform
    hough_radii = np.arange(370, 420, 2)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    # Get image dimensions and center
    img_height, img_width = image_file.shape[:2]
    image_center_x = img_width // 2
    image_center_y = img_height // 2

    # Set a threshold for "centeredness" (adjust as needed)
    threshold = int(img_width * 0.1)  # 10% deviation from center

    circles = []
    for center_y, center_x, radius in zip(cy, cx, radii):
        # If the detected circle is not sufficiently centered, recenter it
        if abs(center_x - image_center_x) > threshold:
            center_x = image_center_x
            center_y = image_center_y
            # Ensure that the circle does not extend past the bottom edge
            if center_y + radius > img_height:
                center_y = img_height - radius
        circles.append((center_y, center_x, radius))

    if len(circles) == 0:
        raise ValueError("No circle detected in the image.")

    # For now, select the first circle candidate. (If needed, add selection criteria here.)
    center_y, center_x, radius = circles[0]

    # Calculate bounding box coordinates
    top_left_x = center_x - radius
    top_left_y = center_y - radius
    # Clamp coordinates to image boundaries
    x_min = max(top_left_x, 0)
    y_min = max(top_left_y, 0)
    x_max = min(center_x + radius, img_width)
    y_max = min(center_y + radius, img_height)

    # Crop the rectangle from the original image and mask
    cropped_img = image_file[y_min:y_max, x_min:x_max]
    cropped_mask = mask_file[y_min:y_max, x_min:x_max]

    return cropped_img, cropped_mask

def roi_crop_images(image_file):
    """
    Crops the circle region of the image.

    Parameters:
      image_file: RGB image (BGR as read by cv2) to be cropped.

    Returns:
      cropped_img: Cropped RGB image.
    """

    # Convert image to grayscale
    gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    # Apply a small Gaussian blur
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    # Edge detection
    edges = cv2.Canny(gray_blurred, 10, 10)

    # Detect circles using the Hough transform
    hough_radii = np.arange(370, 420, 25)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    # Get image dimensions and center
    img_height, img_width = image_file.shape[:2]
    image_center_x = img_width // 2
    image_center_y = img_height // 2

    # Set a threshold for "centeredness" (adjust as needed)
    threshold = int(img_width * 0.1)  # 10% deviation from center

    circles = []
    for center_y, center_x, radius in zip(cy, cx, radii):
        # If the detected circle is not sufficiently centered, recenter it
        if abs(center_x - image_center_x) > threshold:
            center_x = image_center_x
            center_y = image_center_y
            # Ensure that the circle does not extend past the bottom edge
            if center_y + radius > img_height:
                center_y = img_height - radius
        circles.append((center_y, center_x, radius))

    if len(circles) == 0:
        raise ValueError("No circle detected in the image.")

    # For now, select the first circle candidate. (If needed, add selection criteria here.)
    center_y, center_x, radius = circles[0]

    # Calculate bounding box coordinates
    top_left_x = center_x - radius
    top_left_y = center_y - radius
    # Clamp coordinates to image boundaries
    x_min = max(top_left_x, 0)
    y_min = max(top_left_y, 0)
    x_max = min(center_x + radius, img_width)
    y_max = min(center_y + radius, img_height)

    # Crop the rectangle from the original image and mask
    cropped_img = image_file[y_min:y_max, x_min:x_max]

    return cropped_img

def roi_crop_images_masks(image_file, mask_file):
    """
    Crops the circle region of the image.

    Parameters:
      image_file: RGB image (BGR as read by cv2) to be cropped.

    Returns:
      cropped_img: Cropped RGB image.
    """

    # Convert image to grayscale
    gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    # Apply a small Gaussian blur
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    # Edge detection
    edges = cv2.Canny(gray_blurred, 10, 10)

    # Ensure the mask and image have the same dimensions
    if mask_file.shape != image_file.shape:
        raise ValueError(
            f"Dimensions do not match in Mask: {os.path.basename(str(mask_file))} and Image: {os.path.basename(str(image_file))}")

    # Detect circles using the Hough transform
    hough_radii = np.arange(370, 420, 20)
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)

    # Get image dimensions and center
    img_height, img_width = image_file.shape[:2]
    image_center_x = img_width // 2
    image_center_y = img_height // 2

    # Set a threshold for "centeredness" (adjust as needed)
    threshold = int(img_width * 0.1)  # 10% deviation from center

    circles = []
    for center_y, center_x, radius in zip(cy, cx, radii):
        # If the detected circle is not sufficiently centered, recenter it
        if abs(center_x - image_center_x) > threshold:
            center_x = image_center_x
            center_y = image_center_y
            # Ensure that the circle does not extend past the bottom edge
            if center_y + radius > img_height:
                center_y = img_height - radius
        circles.append((center_y, center_x, radius))

    if len(circles) == 0:
        raise ValueError("No circle detected in the image.")

    # For now, select the first circle candidate. (If needed, add selection criteria here.)
    center_y, center_x, radius = circles[0]

    # Calculate bounding box coordinates
    top_left_x = center_x - radius
    top_left_y = center_y - radius
    # Clamp coordinates to image boundaries
    x_min = max(top_left_x, 0)
    y_min = max(top_left_y, 0)
    x_max = min(center_x + radius, img_width)
    y_max = min(center_y + radius, img_height)

    # Crop the rectangle from the original image and mask
    cropped_img = image_file[y_min:y_max, x_min:x_max]
    cropped_mask = mask_file[y_min:y_max, x_min:x_max]


    return cropped_img, cropped_mask

def roi_crop_cv2(image_file):
    """
    Crops the circle region of the image and its corresponding mask,
    detecting the circle via cv2.HoughCircles for speed.

    Parameters:
      image_file: RGB image (BGR as read by cv2) to be cropped.

    Returns:
      cropped_img: Cropped RGB image.
      cropped_mask: Cropped mask.
    """

    # --- 1) Preprocess ---------------------------------------------------
    # Convert to grayscale
    gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    # Gaussian blur to reduce noise
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 1)

    # --- 2) Fast Hough Circle via OpenCV --------------------------------
    img_height, img_width = gray.shape
    # parameters
    dp = 1                                  # accumulator resolution
    minDist = img_height / 8                # min distance between centers
    param1 = 50                             # upper Canny threshold
    param2 = 30                             # accumulator threshold
    minRadius = 370
    maxRadius = 420

    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    if circles is None or len(circles[0]) == 0:
        raise ValueError("No circle detected in the image.")

    # Take the first detected circle (x, y, r)
    x, y, r = circles[0][0]
    center_x, center_y, radius = int(x), int(y), int(r)

    # --- 3) Re-center if off-center ------------------------------------
    image_center_x = img_width // 2
    image_center_y = img_height // 2
    threshold = int(img_width * 0.1)  # allow 10% horizontal deviation

    if abs(center_x - image_center_x) > threshold:
        center_x = image_center_x
        center_y = image_center_y
    # ensure it doesn’t extend past bottom edge
    if center_y + radius > img_height:
        center_y = img_height - radius

    # --- 5) Compute bounding box & crop --------------------------------
    x_min = max(center_x - radius, 0)
    y_min = max(center_y - radius, 0)
    x_max = min(center_x + radius, img_width)
    y_max = min(center_y + radius, img_height)

    cropped_img  = image_file[y_min:y_max, x_min:x_max]

    return cropped_img


def convert_jpg_to_png(name, input_file, output_folder):
    """
    Convert an image file (assumed JPG) to PNG and save in the output folder.
    """
    base_name = os.path.splitext(os.path.basename(name))[0]
    png_file = os.path.join(output_folder, base_name + ".png")

    print("Input shape:", input_file.shape)

    # Save image as PNG (lossless)
    success = cv2.imwrite(png_file, input_file)
    if success:
        print(f"Saved: {png_file}")
    else:
        print(f"Failed to save: {png_file}")


def resample_px_mm_2_classes(image_files, mask_files, threshold=115, target_px_mm=29.98):
    """
    Measures detected circle's "equivalent diameter" after using cropping functions,
    resamples the cropped image and mask to the desired px/mm ratio, and then crops
    the region of interest.

    Parameters:
      image_files: List of paths to RGB images.
      mask_files: List of paths to mask images.
      threshold: Pixel intensity threshold for binarization.
      target_px_mm: Desired pixel-to-mm conversion factor.
    """

    if len(image_files) != len(mask_files):
        raise ValueError("The number of image files and mask files must match.")

    for i in range(len(image_files)):
        # Read image and mask
        image = cv2.imread(image_files[i])
        mask = cv2.imread(mask_files[i])

        # Use the green channel to threshold the image
        green_channel = image[:, :, 1]
        img_height, img_width = green_channel.shape[:2]
        size = (img_height, img_width)

        # Process the mask: map middle values to 90, values above 128 to 255.
        # (Adjust the range if needed.)
        middle_class_range = (41, 84)
        mask_bin = np.zeros_like(mask)
        mask_bin[(mask >= middle_class_range[0]) & (mask <= middle_class_range[1])] = 90
        mask_bin[mask > 128] = 255
        mask_height, mask_width = mask_bin.shape[:2]
        size_mask = (mask_height, mask_width)

        if size != size_mask:
            raise ValueError(
                f"Size mismatch between image {os.path.basename(image_files[i])} and mask {os.path.basename(mask_files[i])}!"
            )

        # Thresholding for particle detection
        ret, thresh1 = cv2.threshold(green_channel, threshold, 255, cv2.THRESH_BINARY_INV)
        label_img = label(thresh1, connectivity=1)
        particles = regionprops(label_img)

        # Filter out small particles (noise)
        min_area = 500
        valid_particles = [p for p in particles if p.area > min_area]

        if valid_particles:
            # Find the particle with the lowest centroid (closest to the top)
            lowest_y = min(p.centroid[0] for p in valid_particles)
            img_height = thresh1.shape[0]
            bottom_particles = [p for p in valid_particles if p.centroid[0] >= lowest_y - 0.01 * img_height]
            bottom_particle = max(bottom_particles, key=lambda p: p.area)
        else:
            print(f"No valid particle found in {os.path.basename(image_files[i])}. Skipping...")
            continue

        # Compute equivalent diameter
        equivalent_diameter = bottom_particle.equivalent_diameter
        current_px_mm = equivalent_diameter / 8  # 8 is a constant from your code; adjust as needed.
        scale_factor = target_px_mm / current_px_mm

        # Use the top of the bounding box of the particle as the top-of-circle
        minr, minc, maxr, maxc = bottom_particle.bbox

        # Crop from the top to just above the detected circle in both image and mask
        roi = crop_from_top_to_circle_rgb(image, minr, margin=0.05)
        roi_mask = crop_from_top_to_circle(mask_bin, minr, margin=0.05)

        # Resize the cropped regions
        if scale_factor < 1:
            interp_img = cv2.INTER_LANCZOS4
            interp_mask = cv2.INTER_NEAREST
        elif scale_factor > 1:
            interp_img = cv2.INTER_CUBIC
            interp_mask = cv2.INTER_NEAREST
        else:
            interp_img = cv2.INTER_LINEAR
            interp_mask = cv2.INTER_NEAREST

        width_crop = int(roi.shape[1] * scale_factor)
        height_crop = int(roi.shape[0] * scale_factor)
        width_crop_mask = int(roi_mask.shape[1] * scale_factor)
        height_crop_mask = int(roi_mask.shape[0] * scale_factor)

        resampled_image = cv2.resize(roi, (width_crop, height_crop), interpolation=interp_img)
        print(f'Image File: {os.path.basename(image_files[i])} --> RESAMPLED')

        resampled_mask = cv2.resize(roi_mask, (width_crop_mask, height_crop_mask), interpolation=interp_mask)
        print(f'Mask File: {os.path.basename(mask_files[i])} --> RESAMPLED')

        # Crop the circular region from the resampled image and mask using roi_crop
        cropped_img, cropped_mask = roi_crop_images_masks(resampled_image, resampled_mask)
        print('Cropping --> SUCCESS')

        convert_jpg_to_png(image_files[i], cropped_img,
                           output_folder=r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\1_class_tloss_FOV_small\pre_processed_imgs")
        print('Image successfully converted')

        convert_jpg_to_png(mask_files[i], cropped_mask,
                           output_folder=r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\1_class_tloss_FOV_small\pre_processed_masks")
        print('Mask successfully converted')

def resample_px_mm_1_class(image_files, mask_files, threshold=115, target_px_mm=29.98):
    """
    Measures detected circle's "equivalent diameter" after using cropping functions,
    resamples the cropped image and mask to the desired px/mm ratio, and then crops
    the region of interest.

    Parameters:
      image_files: List of paths to RGB images.
      mask_files: List of paths to mask images.
      threshold: Pixel intensity threshold for binarization.
      target_px_mm: Desired pixel-to-mm conversion factor.
    """

    if len(image_files) != len(mask_files):
        raise ValueError("The number of image files and mask files must match.")

    for i in range(len(image_files)):
        # Read image and mask
        image = cv2.imread(image_files[i])
        mask = cv2.imread(mask_files[i])

        # Use the green channel to threshold the image
        green_channel = image[:, :, 1]
        img_height, img_width = green_channel.shape[:2]
        size = (img_height, img_width)

        # Process the mask: map middle values to 90, values above 128 to 255.
        mask_bin = np.zeros_like(mask)
        mask_bin[mask > 128] = 255
        mask_height, mask_width = mask_bin.shape[:2]
        size_mask = (mask_height, mask_width)

        if size != size_mask:
            raise ValueError(
                f"Size mismatch between image {os.path.basename(image_files[i])} and mask {os.path.basename(mask_files[i])}!"
            )

        # Thresholding for particle detection
        ret, thresh1 = cv2.threshold(green_channel, threshold, 255, cv2.THRESH_BINARY_INV)
        label_img = label(thresh1, connectivity=1)
        particles = regionprops(label_img)

        # Filter out small particles (noise)
        min_area = 500
        valid_particles = [p for p in particles if p.area > min_area]

        if valid_particles:
            # Find the particle with the lowest centroid (closest to the top)
            lowest_y = min(p.centroid[0] for p in valid_particles)
            img_height = thresh1.shape[0]
            bottom_particles = [p for p in valid_particles if p.centroid[0] >= lowest_y - 0.01 * img_height]
            bottom_particle = max(bottom_particles, key=lambda p: p.area)
        else:
            print(f"No valid particle found in {os.path.basename(image_files[i])}. Skipping...")
            continue

        # Compute equivalent diameter
        equivalent_diameter = bottom_particle.equivalent_diameter
        current_px_mm = equivalent_diameter / 8  # 8 is a constant from your code; adjust as needed.
        scale_factor = target_px_mm / current_px_mm

        # Use the top of the bounding box of the particle as the top-of-circle
        minr, minc, maxr, maxc = bottom_particle.bbox

        # Crop from the top to just above the detected circle in both image and mask
        roi = crop_from_top_to_circle_rgb(image, minr, margin=0.05)
        roi_mask = crop_from_top_to_circle(mask_bin, minr, margin=0.05)

        # Resize the cropped regions
        if scale_factor < 1:
            interp_img = cv2.INTER_LANCZOS4
            interp_mask = cv2.INTER_NEAREST
        elif scale_factor > 1:
            interp_img = cv2.INTER_CUBIC
            interp_mask = cv2.INTER_NEAREST
        else:
            interp_img = cv2.INTER_LINEAR
            interp_mask = cv2.INTER_NEAREST

        width_crop = int(roi.shape[1] * scale_factor)
        height_crop = int(roi.shape[0] * scale_factor)
        width_crop_mask = int(roi_mask.shape[1] * scale_factor)
        height_crop_mask = int(roi_mask.shape[0] * scale_factor)

        resampled_image = cv2.resize(roi, (width_crop, height_crop), interpolation=interp_img)
        print(f'Image File: {os.path.basename(image_files[i])} --> RESAMPLED')

        resampled_mask = cv2.resize(roi_mask, (width_crop_mask, height_crop_mask), interpolation=interp_mask)
        print(f'Mask File: {os.path.basename(mask_files[i])} --> RESAMPLED')

        # Crop the circular region from the resampled image and mask using roi_crop
        cropped_img, cropped_mask = roi_crop_images_masks(resampled_image, resampled_mask)
        print('Cropping --> SUCCESS')

        convert_jpg_to_png(image_files[i], cropped_img,
                           output_folder=r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\1_class_tloss_FOV_small\pre_processed_imgs")
        print('Image successfully converted')

        convert_jpg_to_png(mask_files[i], cropped_mask,
                           output_folder=r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Prebuilt NNs\MSR-NET\Ricc\1_class_tloss_FOV_small\pre_processed_masks")
        print('Mask successfully converted')


# --------------------------- Directories -----------------------

original_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Images\RGB_images"
mask_path = r"C:\Users\ricca\OneDrive\Work\01_Projects\2024\Image Processing Algorithm\Images\Mask_1"
image_files = sorted(glob(os.path.join(original_path, "*.jpg")))  # Adjust extension if needed
mask_files = sorted(glob(os.path.join(mask_path, "*.jpg")))  # Adjust extension if needed

# ------------------------------ Run Script ----------------------------
resample_px_mm_1_class(image_files, mask_files, threshold=115, target_px_mm=29.98)
