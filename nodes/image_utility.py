from email.mime import image
import os
import io
from turtle import width
import requests
import json
import numpy as np
import cv2
from io import BytesIO
from PIL import Image, ImageOps, ImageSequence, ImageFile, ImageDraw
from PIL.PngImagePlugin import PngInfo
from sympy import prime

import folder_paths
import comfy.utils
from comfy.cli_args import args
from datetime import datetime
from comfy.utils import ProgressBar

import torch
import torch.nn.functional as NNF
from torchvision import transforms

import base64
import sys
from comfy.comfy_types.node_typing import IO

# tensor [b,c,h,w]
# pil [h,w,c]
# np [h,w,c]


def pil2tensor_mask(pil):
    output_images = []
    output_masks = []
    for i in ImageSequence.Iterator(pil):
        i = ImageOps.exif_transpose(i)
        if i.mode == "I":
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        if "A" in i.getbands():
            mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return (output_image, output_mask)


toPIL = transforms.ToPILImage()
toTensor = transforms.ToTensor()


def pil2tensor(pil):
    # [C, H, W] to [H, W, C]
    return toTensor(pil).permute(1, 2, 0)


def tensor2pil(tensor):
    # [H, W, C] to [C, H, W]
    return toPIL(tensor.permute(2, 0, 1))


def tensor2batch(tensor, h, w, c):
    tensor = torch.cat(tensor)
    tensor = tensor.reshape(-1, h, w, c)
    return tensor


def batch2list(tensor_batch):
    tensors = []
    for i in range(tensor_batch.shape[0]):
        tensors.append(tensor_batch[i])
    return tensors


def list2batch(tensor_list):
    return torch.cat(tensor_list)


def rgba2rgb(pil):
    bg = Image.new("RGB", pil.size, (255, 255, 255))
    bg.paste(pil, pil)
    pil = bg


def load_pil_from_url(url):
    response = requests.get(url)
    pil = Image.open(BytesIO(response.content))
    name = url.split("/")[-1]
    return pil, name


class DownloadImageNode:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (IO.IMAGE,),
                "filename_prefix": (IO.STRING, {"default": "ComfyUI_%time%_%batch_num%"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "function"
    OUTPUT_NODE = True
    CATEGORY = "Fair/image"

    def function(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = []
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))

            now = datetime.now()
            timename = now.strftime("%Y-%m-%d-%H-%M-%S")
            filename_with_batch_num = filename_with_batch_num.replace("%time%", timename)
            file = f"{filename_with_batch_num}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1

        return {"ui": {"images": results}}


class SaveImageToDirectoryNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IO.IMAGE, {"defaultInput": True}),
                "directory": (IO.STRING, {"defaultInput": True}),
                "name": (IO.STRING, {"defaultInput": True}),
                "type": (["png", "jpg"], {"default": "png"}),
                "compress_level": (IO.INT, {"default": 4, "min": 0, "max": 9}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "function"
    OUTPUT_NODE = True
    CATEGORY = "Fair/image"

    def function(self, image, directory, name, type, compress_level):
        for i in image:
            pil = tensor2pil(i)
            name = f"{name}.{type}"
            pil.save(os.path.join(directory, name), compress_level=compress_level)

        return ()


class ImageResizeNode:
    def __init__(self):
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IO.IMAGE,),
                "resize_to": (IO.INT, {"default": 1024, "max": 4096}),
                "side": (["shortest", "longest", "width", "height"], {"default": "longest"}),
                "interpolation": (["lanczos", "nearest", "bilinear", "bicubic", "area", "nearest-exact"], {"default": "lanczos"}),
            }
        }

    RETURN_TYPES = (IO.IMAGE, IO.INT, IO.INT)
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "node_function"
    OUTPUT_NODE = True
    CATEGORY = "Fair/image"

    def node_function(self, image, resize_to, side, interpolation):
        image = image.movedim(-1, 1)

        image_height, image_width = image.shape[-2:]

        longer_side = "height" if image_height > image_width else "width"
        shorter_side = "height" if image_height < image_width else "width"

        height, width = 0, 0

        if side == "shortest":
            side = shorter_side
        elif side == "longest":
            side = longer_side

        if side == "width":
            width = resize_to
            height = image_height * (resize_to / image_width)
        elif side == "height":
            width = image_width * (resize_to / image_height)
            height = resize_to

        width = int(width)
        height = int(height)

        image = comfy.utils.common_upscale(image, width, height, interpolation, "center")
        image = image.movedim(1, -1)

        return (image, width, height)


class VideoToImageNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": (IO.STRING, {"defaultInput": False}),
                "capture_rate": (IO.INT, {"default": 30}),
                "frame_offset": (IO.INT, {"default": 0}),
                "image_dir": (IO.STRING, {"defaultInput": False}),
                "image_name_prefix": (IO.STRING, {"defaultInput": False}),
            }
        }

    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("image_paths",)
    FUNCTION = "node_function"
    OUTPUT_NODE = True
    CATEGORY = "Fair/image"

    def node_function(self, video_path, capture_rate, frame_offset, image_dir, image_name_prefix):
        video_path = video_path.replace('"', "")
        image_dir = image_dir.replace('"', "")

        vc = cv2.VideoCapture(video_path)
        width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = vc.get(cv2.CAP_PROP_FPS)
        frame_count = vc.get(cv2.CAP_PROP_FRAME_COUNT)
        print("width:{} \nheight:{} \nfps:{} \nnum_frames:{}".format(width, height, fps, frame_count))

        progress_total = frame_count
        progress_bar = ProgressBar(progress_total)

        frame_counter = 1 - frame_offset
        image_paths = []
        if vc.isOpened():
            while True:
                frame_success, frame_image = vc.read()
                if frame_success:
                    if frame_counter % capture_rate == 0:
                        file_name = str(frame_counter) + ".jpg"
                        file_name = image_name_prefix + file_name
                        image_path = os.path.join(image_dir, file_name)
                        image_paths.append(image_path)
                        cv2.imwrite(image_path, frame_image)
                else:
                    break
                frame_counter += 1
                progress_bar.update_absolute(frame_counter, progress_total)
        vc.release()

        return (image_paths,)


class ImageToVideoNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_dir": (IO.STRING, {"defaultInput": False}),
                "frame_rate": (IO.INT, {"default": 30}),
                "video_dir": (IO.STRING, {"defaultInput": False}),
                "video_name": (IO.STRING, {"defaultInput": False, "default": "output"}),
            }
        }

    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("image_paths", "video_path")
    FUNCTION = "node_function"
    OUTPUT_NODE = True
    CATEGORY = "Fair/image"

    def node_function(self, image_dir, frame_rate, video_dir, video_name):
        image_dir = image_dir.replace('"', "")
        video_dir = video_dir.replace('"', "")
        video_path = os.path.join(video_dir, f"{video_name}.mp4")

        image_paths = []
        image_name_list = os.listdir(image_dir)
        for image_name in image_name_list:
            image_paths.append(os.path.join(image_dir, image_name))

        progress_total = len(image_name_list)
        progress_bar = ProgressBar(progress_total)

        img0 = cv2.imread(image_paths[0])
        height, width, layers = img0.shape
        print(f"video resolution:{width}*{height}")

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

        progress_counter = 0
        for image_path in image_paths:
            img = cv2.imread(image_path)
            video_writer.write(img)

            progress_counter += 1
            progress_bar.update_absolute(progress_counter, progress_total)

        video_writer.release()

        return (image_paths, video_path)


class LoadImageFromURLNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": (IO.STRING, {"defaultInput": False}),
                "channels": (["RGB", "RGBA"], {"default": "RGB"}),
            }
        }

    RETURN_TYPES = (IO.IMAGE, IO.MASK, IO.STRING)
    RETURN_NAMES = ("image", "mask", "name")
    FUNCTION = "node_function"
    CATEGORY = "Fair/image"

    def node_function(self, url, channels):
        pil, name = load_pil_from_url(url)
        pil.convert(channels)
        img_out = pil2tensor(pil)
        return (img_out, img_out, name)


def load_image_to_tensor(directory, recursive, channels):
    image_file_paths = []
    if recursive:
        for root, _, files in os.walk(directory):
            for file_name in files:
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_file_paths.append(os.path.join(root, file_name))
    else:
        for file_name in os.listdir(directory):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_file_paths.append(os.path.join(directory, file_name))

    image_tensors = []
    image_dirs = []
    image_names = []

    progress_bar = ProgressBar(image_file_paths.__len__())

    for image_path in image_file_paths:

        with Image.open(image_path) as pil:
            pil = ImageOps.exif_transpose(pil)  # Handle EXIF orientation

            if pil.mode == "RGBA" and channels == "RGB":
                pil = rgba2rgb(pil)
            elif pil.mode == "RGB" and channels == "RGBA":
                pil = pil.convert("RGBA")

            image_tensor = pil2tensor(pil)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            image_tensors.append(image_tensor)

            image_dirs.append(os.path.dirname(image_path))
            file_name = os.path.basename(image_path)
            image_names.append(os.path.splitext(file_name)[0])  # Remove file extension

            progress_bar.update(1)

    return (image_tensors, image_dirs, image_names)


class LoadImageFromDirectoryNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": (IO.STRING, {"default": "", "forceInput": False}),
                "recursive": (IO.BOOLEAN, {"default": False}),
                "channels": (["RGB", "RGBA"], {"default": "RGB"}),
            }
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/image"

    RETURN_TYPES = (IO.IMAGE, IO.STRING, IO.STRING)
    RETURN_NAMES = ("image", "directory", "name")
    OUTPUT_IS_LIST = (True, True, True)

    def node_function(self, directory, recursive, channels):
        if not directory or not os.path.isdir(directory):
            raise Exception("folder_path is not valid: " + directory)

        (out_image, out_dir, out_name) = load_image_to_tensor(directory, recursive, channels)

        return (out_image, out_dir, out_name)


class LoadImageBatchFromDirectoryNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": (IO.STRING, {"default": "", "forceInput": False}),
                "recursive": (IO.BOOLEAN, {"default": False}),
                "channels": (["RGB", "RGBA"], {"default": "RGB"}),
            }
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/image"

    RETURN_TYPES = (IO.IMAGE, IO.STRING, IO.STRING)
    RETURN_NAMES = ("images", "directory", "name")
    OUTPUT_IS_LIST = (False, True, True)

    def node_function(self, directory, recursive, channels):
        if not directory or not os.path.isdir(directory):
            raise Exception("folder_path is not valid: " + directory)

        (out_image, out_dir, out_name) = load_image_to_tensor(directory, recursive, channels)
        out_image = list2batch(out_image)

        return (out_image, out_dir, out_name)


class FillAlphaNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IO.IMAGE, {"default": "", "forceInput": True}),
                "alpha_threshold": (IO.INT, {"default": "128"}),
                "r": (IO.INT, {"default": "0"}),
                "g": (IO.INT, {"default": "128"}),
                "b": (IO.INT, {"default": "0"}),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    FUNCTION = "node_function"
    CATEGORY = "Fair/image"

    def fill_alpha(self, tensor, alpha_threshold, fill_color):

        pil = tensor2pil(tensor)
        if pil.mode != "RGBA":
            raise Exception("Image mode is not RGBA")

        pixels = pil.getdata()

        new_pixels = []
        for pixel in pixels:
            r, g, b, a = pixel
            if a < alpha_threshold:
                new_pixels.append(fill_color)
            else:
                new_pixels.append((int(r * a / 255.0), int(g * a / 255.0), int(b * a / 255.0)))

        new_pil = Image.new("RGB", pil.size)
        new_pil.putdata(new_pixels)
        image_tensor = pil2tensor(new_pil)
        return image_tensor

    def node_function(self, image, alpha_threshold, r, g, b):
        tensor_filled = self.fill_alpha(image, alpha_threshold, (r, g, b))
        return (tensor_filled,)


def pil_to_base64(pli_image, pnginfo=None, header=False):
    # 创建一个BytesIO对象，用于临时存储图像数据
    image_data = io.BytesIO()

    # 将图像保存到BytesIO对象中，格式为PNG
    pli_image.save(image_data, format="PNG", pnginfo=pnginfo)

    # 将BytesIO对象的内容转换为字节串
    image_data_bytes = image_data.getvalue()

    # 将图像数据编码为Base64字符串
    encoded_image = "data:image/png;base64," + base64.b64encode(image_data_bytes).decode("utf-8") if header is True else base64.b64encode(image_data_bytes).decode("utf-8")

    return encoded_image


class ImageToBase64Node:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IO.IMAGE, {"defaultInput": True}),
            }
        }

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "function"
    OUTPUT_NODE = True
    CATEGORY = "Fair/image"

    def function(self, image):
        pil = tensor2pil(image)
        encoded_base64 = pil_to_base64(pil)
        return (encoded_base64,)


def base64_to_pil(base64_string):
    # 去除前缀
    base64_list = base64_string.split(",", 1)
    if len(base64_list) == 2:
        prefix, base64_data = base64_list
    else:
        base64_data = base64_list[0]

    # 从base64字符串中解码图像数据
    image_data = base64.b64decode(base64_data)

    # 创建一个内存流对象
    image_stream = io.BytesIO(image_data)

    # 使用PIL的Image模块打开图像数据
    image = Image.open(image_stream)

    return image


class Base64ToImageNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": (IO.STRING, {"defaultInput": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "function"
    OUTPUT_NODE = True
    CATEGORY = "Fair/image"

    def function(self, string):
        pil = base64_to_pil(string)
        image = pil2tensor(pil)
        return (image,)


class OutpaintingPadNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IO.IMAGE, {"defaultInput": True, "forceInput": True}),
                "target_width": (IO.INT, {"default": 1024, "min": 64, "max": 8192}),
                "target_height": (IO.INT, {"default": 1024, "min": 64, "max": 8192}),
                "align": (["left", "left-top", "top", "top-right", "right", "right-bottom", "bottom", "left-bottom", "center"], {"default": "center"}),
            }
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/image"
    RETURN_TYPES = (IO.INT, IO.INT, IO.INT, IO.INT)
    RETURN_NAMES = ("left", "top", "right", "bottom")

    def node_function(self, image, target_width, target_height, align):
        image_height = image.shape[1]
        image_width = image.shape[2]

        if image_width > target_width or image_height > target_height:
            raise Exception(f"Image size is larger than target size\nimage size: {image_width}x{image_height}, target size: {target_width}x{target_height}")

        # aligns
        if align == "left":
            left = 0
            top = (target_height - image_height) // 2
            right = target_width - image_width
            bottom = target_height - image_height - top
        elif align == "left-top":
            left = 0
            top = 0
            right = target_width - image_width
            bottom = target_height - image_height
        elif align == "top":
            left = (target_width - image_width) // 2
            top = 0
            right = target_width - image_width - left
            bottom = target_height - image_height - top
        elif align == "top-right":
            left = target_width - image_width
            top = 0
            right = 0
            bottom = target_height - image_height
        elif align == "right":
            left = target_width - image_width
            top = (target_height - image_height) // 2
            right = 0
            bottom = target_height - image_height - top
        elif align == "right-bottom":
            left = target_width - image_width
            top = target_height - image_height
            right = 0
            bottom = 0
        elif align == "bottom":
            left = (target_width - image_width) // 2
            top = target_height - image_height
            right = target_width - image_width - left
            bottom = 0
        elif align == "left-bottom":
            left = 0
            top = target_height - image_height
            right = target_width - image_width
            bottom = 0
        elif align == "center":
            left = (target_width - image_width) // 2
            top = (target_height - image_height) // 2
            right = target_width - image_width - left
            bottom = target_height - image_height - top

        return (left, top, right, bottom)


class ImageSizeNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IO.IMAGE, {"defaultInput": True, "forceInput": True}),
            }
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/image"
    RETURN_TYPES = (IO.INT, IO.INT, IO.INT, IO.INT, "FLOAT")
    RETURN_NAMES = ("width", "height", "max_side", "min_side", "aspect_ratio")

    def node_function(self, image):
        height = image.shape[1]
        width = image.shape[2]
        max_side = max(width, height)
        min_side = min(width, height)
        aspect_ratio = width / height

        return (width, height, max_side, min_side, aspect_ratio)


class ImagesRangeNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (IO.IMAGE, {"defaultInput": True}),
                "start": (IO.INT, {"default": 0, "min": -sys.maxsize - 1, "max": sys.maxsize}),
                "use_start": ("BOOL", {"default": False}),
                "end": (IO.INT, {"default": -1, "min": -sys.maxsize - 1, "max": sys.maxsize}),
                "use_end": ("BOOL", {"default": False}),
            }
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/image"
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("images",)

    def node_function(self, images, start, use_start, end, use_end):
        images = images[start if use_start else None : end if use_end else None]
        return (images,)


class ImagesIndexNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (IO.IMAGE, {"defaultInput": True}),
                "index": (IO.INT, {"default": -1, "min": -sys.maxsize - 1, "max": sys.maxsize}),
            }
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/image"
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)

    def node_function(self, images, index):
        return (images[index].unsqueeze(0),)


class ImagesCatNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (IO.IMAGE, {"defaultInput": True}),
                "images_cat": (IO.IMAGE, {"defaultInput": True}),
            }
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/image"
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("images",)

    def node_function(self, images, images_cat):
        return (torch.cat((images, images_cat), dim=0),)
