import os
from PIL import Image, ImageOps
import torch
import numpy as np
import itertools
from comfy.utils import common_upscale, ProgressBar
from comfy.k_diffusion.utils import FolderOfImages
from typing import Iterable
from torchvision import transforms


def validate_load_images(directory: str):
    if not os.path.isdir(directory):
        return f"Directory '{directory}' cannot be found."
    dir_files = os.listdir(directory)
    if len(dir_files) == 0:
        return f"No files in directory '{directory}'."

    return True


def strip_path(path):
    # This leaves whitespace inside quotes and only a single "
    # thus ' ""test"' -> '"test'
    # consider path.strip(string.whitespace+"\"")
    # or weightier re.fullmatch("[\\s\"]*(.+?)[\\s\"]*", path).group(1)
    path = path.strip()
    if path.startswith('"'):
        path = path[1:]
    if path.endswith('"'):
        path = path[:-1]
    return path


def get_sorted_dir_files_from_directory(
    directory: str,
    skip_first_images: int = 0,
    select_every_nth: int = 1,
    extensions: Iterable = None,
):
    directory = strip_path(directory)
    dir_files = os.listdir(directory)
    dir_files = sorted(dir_files)
    dir_files = [os.path.join(directory, x) for x in dir_files]
    dir_files = list(filter(lambda filepath: os.path.isfile(filepath), dir_files))
    # filter by extension, if needed
    if extensions is not None:
        extensions = list(extensions)
        new_dir_files = []
        for filepath in dir_files:
            ext = "." + filepath.split(".")[-1]
            if ext.lower() in extensions:
                new_dir_files.append(filepath)
        dir_files = new_dir_files
    # start at skip_first_images
    dir_files = dir_files[skip_first_images:]
    dir_files = dir_files[0::select_every_nth]
    return dir_files


def images_generator(
    directory: str,
    image_load_cap: int = 0,
    skip_first_images: int = 0,
    select_every_nth: int = 1,
    meta_batch=None,
    unique_id=None,
):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory} cannot be found.")
    dir_files = get_sorted_dir_files_from_directory(directory, skip_first_images, select_every_nth, FolderOfImages.IMG_EXTENSIONS)

    if len(dir_files) == 0:
        raise FileNotFoundError(f"No files in directory '{directory}'.")
    if image_load_cap > 0:
        dir_files = dir_files[:image_load_cap]
    sizes = {}
    has_alpha = False
    for image_path in dir_files:
        i = Image.open(image_path)
        # exif_transpose can only ever rotate, but rotating can swap width/height
        i = ImageOps.exif_transpose(i)
        has_alpha |= "A" in i.getbands()
        count = sizes.get(i.size, 0)
        sizes[i.size] = count + 1
    size = max(sizes.items(), key=lambda x: x[1])[0]
    yield size[0], size[1], has_alpha
    if meta_batch is not None:
        yield min(image_load_cap, len(dir_files)) or len(dir_files)

    iformat = "RGBA" if has_alpha else "RGB"

    def load_image(file_path):
        i = Image.open(file_path)
        i = ImageOps.exif_transpose(i)
        i = i.convert(iformat)
        i = np.array(i, dtype=np.float32)
        # This nonsense provides a nearly 50% speedup on my system
        torch.from_numpy(i).div_(255)
        if i.shape[0] != size[1] or i.shape[1] != size[0]:
            i = torch.from_numpy(i).movedim(-1, 0).unsqueeze(0)
            i = common_upscale(i, size[0], size[1], "lanczos", "center")
            i = i.squeeze(0).movedim(0, -1).numpy()
        if has_alpha:
            i[:, :, -1] = 1 - i[:, :, -1]
        return i

    total_images = len(dir_files)
    processed_images = 0
    pbar = ProgressBar(total_images)
    images = map(load_image, dir_files)
    try:
        prev_image = next(images)
        while True:
            next_image = next(images)
            yield prev_image
            processed_images += 1
            pbar.update_absolute(processed_images, total_images)
            prev_image = next_image
    except StopIteration:
        pass
    if meta_batch is not None:
        meta_batch.inputs.pop(unique_id)
        meta_batch.has_closed_inputs = True
    if prev_image is not None:
        yield prev_image


def load_images_func(
    folder: str,
    image_load_cap: int = 0,
    skip_first_images: int = 0,
    select_every_nth: int = 1,
    meta_batch=None,
    unique_id=None,
):
    if meta_batch is None or unique_id not in meta_batch.inputs:
        gen = images_generator(
            folder,
            image_load_cap,
            skip_first_images,
            select_every_nth,
            meta_batch,
            unique_id,
        )
        (width, height, has_alpha) = next(gen)
        if meta_batch is not None:
            meta_batch.inputs[unique_id] = (gen, width, height, has_alpha)
            meta_batch.total_frames = min(meta_batch.total_frames, next(gen))
    else:
        gen, width, height, has_alpha = meta_batch.inputs[unique_id]

    if meta_batch is not None:
        gen = itertools.islice(gen, meta_batch.frames_per_batch)
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3 + has_alpha)))))

    if has_alpha:
        images = images[:, :, :, :3]
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded from directory '{folder}'.")
    image_paths = get_sorted_dir_files_from_directory(folder, skip_first_images, select_every_nth, FolderOfImages.IMG_EXTENSIONS)

    image_name_list = [os.path.basename(x) for x in image_paths]
    image_name_no_postfix_list = [os.path.splitext(x)[0] for x in image_name_list]
    return (images, folder, image_name_no_postfix_list)


class LoadImageFromFolderNode:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"folder": ("STRING", {"default": "", "forceInput": False})}}

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "FOLDER", "NAME")
    FUNCTION = "function"
    CATEGORY = "Fair/image"

    def function(self, folder: str, **kwargs):
        folder = strip_path(folder)
        if folder is None or validate_load_images(folder) != True:
            raise Exception("directory is not valid: " + folder)

        return load_images_func(folder, **kwargs)


def load_images_to_tensors(folder_path, recursive, convert_to_rgb):
    image_file_paths = []
    if recursive:
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_file_paths.append(os.path.join(root, file_name))
    else:
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_file_paths.append(os.path.join(folder_path, file_name))

    image_tensors = []
    image_nps = []
    images = []
    max_w, max_h = 0, 0

    for image_path in image_file_paths:
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        w, h = image.size
        max_w = max(max_w, w)
        max_h = max(max_h, h)
        images.append(image)

    for image in images:
        if image.size[0] != max_w or image.size[1] != max_h:
            image = image.resize((max_w, max_h), Image.LANCZOS)

        if convert_to_rgb:
            image = image.convert("RGB")

        image_np = np.array(image, dtype=np.float32)
        image_nps.append(image_np)
        image_tensor = torch.from_numpy(image_np).div(255.0)
        image_tensors.append(image_tensor)

    image_tensors = torch.cat(image_tensors)
    image_tensors = image_tensors.reshape(-1, max_h, max_w, 3)

    # names
    image_name_list = [os.path.basename(x) for x in image_file_paths]
    image_names = [os.path.splitext(x)[0] for x in image_name_list]
    image_folders = [os.path.dirname(p) for p in image_file_paths]

    return image_tensors, image_folders, image_names


class LoadImagesFromFolderNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "", "forceInput": False}),
                "recursive": ("BOOLEAN", {"default": False}),
                "convert_to_rgb": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "FOLDER", "NAME")
    FUNCTION = "node_function"
    CATEGORY = "Fair/image"

    def node_function(self, folder_path, recursive, convert_to_rgb):
        if not folder_path or not os.path.isdir(folder_path):
            raise Exception("folder_path is not valid: " + folder_path)

        out_images, out_folders, out_image_names = load_images_to_tensors(folder_path, recursive, convert_to_rgb)

        return (out_images, out_folders, out_image_names)
