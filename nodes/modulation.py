import numpy as np
import concurrent.futures


def modulation(image_np, index, direction, speed):
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
