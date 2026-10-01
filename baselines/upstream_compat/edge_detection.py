
import cv2
import numpy as np


def ndarray_add_num(array, maximum, value):
    """Add a scalar with explicit saturation.

    The published artifact imports this helper from an omitted ``edge``
    package. Its call sites pass ``(image, 255, 10)``; using a wider signed
    dtype avoids uint8 wraparound and preserves the intended capped addition.
    """

    return np.clip(
        array.astype(np.int16) + int(value), 0, int(maximum)
    ).astype(array.dtype)


def ndarray_add_arr(left, right, maximum=255):
    """Add two image arrays with explicit saturation."""

    return np.clip(
        left.astype(np.int16) + right.astype(np.int16), 0, int(maximum)
    ).astype(left.dtype)


def zero_crossing_edge_detection(img):
    # Apply Gaussian Blur
    # img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    # Apply Laplacian to find zero crossings
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    # Detect zero-crossings
    _, edges = cv2.threshold(np.abs(laplacian), 0, 255, cv2.THRESH_BINARY)
    return edges


def fined_edge_detection(image, kernel_type="extend"):
    import time
    start = time.process_time()

    # Apply the original edge-detection operation.
    kernel_extend = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]])

    kernel_simple = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]])

    kernel_extend_5x5 = np.array([[-1, -1, -1, -1, -1],
                                  [-1, -1, -1, -1, -1],
                                  [-1, -1, 24, -1, -1],
                                  [-1, -1, -1, -1, -1],
                                  [-1, -1, -1, -1, -1]])

    kernel_scharr_x = np.array([[-3, 0, 3],
                                [-10, 0, 10],
                                [-3, 0, 3]])

    kernel_scharr_y = np.array([[-3, -10, -3],
                                [0, 0, 0],
                                [3, 10, 3]])

    kernel_choice = {
        'simple': kernel_simple,
        'extend': kernel_extend,
        'extend_5x5': kernel_extend_5x5
    }

    # Apply the original edge-detection operation.
    sharpened = cv2.filter2D(image, -1, kernel_choice[kernel_type])

    _, binary_image = cv2.threshold(sharpened, 10, 255, cv2.THRESH_BINARY)

    # Apply the original edge-detection operation.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # Apply the original edge-detection operation.
    min_size = 50  # Apply the original edge-detection operation.
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            binary_image[labels == i] = 0

    # Apply the original edge-detection operation.
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Apply the original edge-detection operation.
    h, w = binary_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    for cnt in contours:
        # Apply the original edge-detection operation.
        x, y, w, h = cv2.boundingRect(cnt)
        seed_point = (x, y)

        # Apply the original edge-detection operation.
        cv2.floodFill(binary_image, mask, seed_point, 255)

    # Apply the original edge-detection operation.
    binary_image_inv = cv2.bitwise_not(binary_image)
    enhanced_image = ndarray_add_num(sharpened, 255, 10)
    final_image = ndarray_add_arr(ndarray_add_num(enhanced_image, 255, 10), binary_image_inv)
    # final_image = cv2.bitwise_not(final_image)

    end = time.process_time()
    print("==== detect_sep_lines_with_lsd Completed in %.3f s " % (end - start))
    return final_image
