import numpy as np
import concurrent.futures


def modulation(image_np, index, direction, speed):
    # Make a copy to avoid modifying the original
    image_np = image_np.copy()
    h, w = image_np.shape

    if direction == "up_to_down":
        for x in range(w):
            for y in range(h):
                old = image_np[y, x]
                new = np.round(old)
                image_np[y, x] = new
                error = old - new
                if y == 0:
                    error -= index * speed
                if y + 1 < h:
                    image_np[y + 1, x] += error
    elif direction == "down_to_up":
        for x in range(w):
            for y in range(h - 1, -1, -1):
                old = image_np[y, x]
                new = np.round(old)
                image_np[y, x] = new
                error = old - new
                if y == h - 1:
                    error -= index * speed
                if y - 1 >= 0:
                    image_np[y - 1, x] += error
    elif direction == "left_to_right":
        for y in range(h):
            for x in range(w):
                old = image_np[y, x]
                new = np.round(old)
                image_np[y, x] = new
                error = old - new
                if x == 0:
                    error -= index * speed
                if x + 1 < w:
                    image_np[y, x + 1] += error
    elif direction == "right_to_left":
        for y in range(h):
            for x in range(w - 1, -1, -1):
                old = image_np[y, x]
                new = np.round(old)
                image_np[y, x] = new
                error = old - new
                if x == w - 1:
                    error -= index * speed
                if x - 1 >= 0:
                    image_np[y, x - 1] += error

    return image_np


def floyd_steinberg(image_np):
    h, w = image_np.shape
    for y in range(h):
        for x in range(w):
            old = image_np[y, x]
            new = np.round(old)
            image_np[y, x] = new
            error = old - new
            # precomputing the constants helps
            if x + 1 < w:
                image_np[y, x + 1] += error * 0.4375  # right, 7 / 16
            if (y + 1 < h) and (x + 1 < w):
                image_np[y + 1, x + 1] += error * 0.0625  # right, down, 1 / 16
            if y + 1 < h:
                image_np[y + 1, x] += error * 0.3125  # down, 5 / 16
            if (x - 1 >= 0) and (y + 1 < h):
                image_np[y + 1, x - 1] += error * 0.1875  # left, down, 3 / 16
    return image_np


def process_modulation(images_np, direction, speed):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(modulation, images_np, range(len(images_np)), [direction] * len(images_np), [speed] * len(images_np))
    return list(results)


def modulation_worker(args):
    """Helper function for ProcessPoolExecutor with error handling"""
    try:
        img_np, idx, direction, speed = args
        # Ensure we're working with a copy
        img_np = np.array(img_np, copy=True)
        return modulation(img_np, idx, direction, speed)
    except Exception as e:
        print(f"Error in modulation_worker for index {args[1]}: {e}")
        # Return original image on error
        return args[0]


def process_modulation_ProcessPool(images_np, direction, speed):
    """Process images using ProcessPool with proper error handling and resource limits"""

    # Prepare arguments - ensure numpy arrays are properly copied
    arg_list = [(img_np.copy(), idx, direction, speed) for idx, img_np in enumerate(images_np)]

    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Add timeout to prevent hanging
            futures = [executor.submit(modulation_worker, args) for args in arg_list]
            results = []

            for future in concurrent.futures.as_completed(futures, timeout=300):  # 5 minutes timeout
                try:
                    result = future.result(timeout=60)  # 1 minute per image
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    print("Timeout error processing image, using original")
                    results.append(arg_list[len(results)][0])
                except Exception as e:
                    print(f"Error processing image: {e}, using original")
                    results.append(arg_list[len(results)][0])

            return results
    except Exception as e:
        print(f"ProcessPool error: {e}, falling back to ThreadPool")
        # Fallback to ThreadPool if ProcessPool fails
        return process_modulation(images_np, direction, speed)
