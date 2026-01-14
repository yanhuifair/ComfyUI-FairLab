from calendar import c
import os
import io
import requests
import json
import numpy as np
import cv2
from io import BytesIO
from PIL import Image, ImageOps, ImageSequence, ImageFile, ImageDraw
from PIL.PngImagePlugin import PngInfo

import folder_paths
import comfy.utils
from comfy.cli_args import args
from datetime import datetime
from comfy.utils import ProgressBar

import torch
import torch.nn.functional as NNF
from torchvision import transforms
import torchvision.transforms.functional as f

import base64
import sys
from comfy.comfy_types.node_typing import IO

from .modulation import process_modulation, process_modulation_ProcessPool
import matplotlib.pyplot as plt
from perfect_pixel import get_perfect_pixel

# tensor [B,C,H,W]
# pil [H,W,C]
# np [H,W,C]
# comfyui image [B,H,W,C]


def tensor_to_cv2(tensor):
    tensor = tensor.permute(1, 2, 0)  # [C,H,W] to [H,W,C]
    array = tensor.cpu().numpy()
    array = (array * 255).astype(np.uint8)
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array


def cv2_to_tensor(array):
    array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    array = array.astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor


def pil_to_tensor_mask(pil):
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


def pil_to_tensor(pil):
    # [C, H, W] to [H, W, C]
    return toTensor(pil).permute(1, 2, 0)


def tensor_to_pil(tensor):
    if len(tensor.shape) == 2:
        # [H, W] to [H, W]
        return toPIL(tensor)
    else:
        # [H, W, C] to [C, H, W]
        return toPIL(tensor.permute(2, 0, 1))


def tensor_to_batch(tensor, h, w, c):
    tensor = torch.cat(tensor)
    tensor = tensor.reshape(-1, h, w, c)
    return tensor


def batch_to_list(tensor_batch):
    tensors = []
    for i in range(tensor_batch.shape[0]):
        tensors.append(tensor_batch[i])
    return tensors


def list_to_batch(tensor_list):
    return torch.cat(tensor_list)


def tensor_list_to_batch(tensors):
    return torch.stack(tensors, dim=0)


def rgba_to_rgb(pil):
    bg = Image.new("RGB", pil.size, (255, 255, 255))
    bg.paste(pil, pil)
    pil = bg


def load_pil_from_url(url):
    response = requests.get(url)
    pil = Image.open(BytesIO(response.content))
    name = url.split("/")[-1]
    return pil, name


def pil_to_np(pilimage):
    return np.array(pilimage) / 255


def np_to_pil(image):
    return Image.fromarray((image * 255).astype("uint8"))


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
                "filename_prefix": (IO.STRING, {"default": "ComfyUI_{time}_{batch_num}"}),
            },
            "optional": {
                "masks": (IO.MASK,),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "node_function"
    OUTPUT_NODE = True
    CATEGORY = "Fair/image"

    def node_function(self, images, masks=None, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        if masks is not None:
            filename_prefix += self.prefix_append
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
            results = []
            for batch_number, image, mask in zip(range(len(images)), images, masks):
                i = 255.0 * image.cpu().numpy()
                # mask
                alpha = 1 - mask
                a = 255.0 * alpha.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                # alpha
                a_resized = Image.fromarray(a).resize(img.size, Image.LANCZOS)
                a_resized = np.clip(a_resized, 0, 255).astype(np.uint8)
                img.putalpha(Image.fromarray(a_resized, mode="L"))

                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                filename_with_batch_num = filename.replace("{batch_num}", str(batch_number))

                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
                filename_with_batch_num = filename_with_batch_num.replace("{time}", timestamp)
                file_name = f"{filename_with_batch_num}.png"
                img.save(
                    os.path.join(full_output_folder, file_name),
                    pnginfo=metadata,
                    compress_level=self.compress_level,
                )
                results.append({"filename": file_name, "subfolder": subfolder, "type": self.type})
                counter += 1

            return {"ui": {"images": results}}
        else:
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

                filename_with_batch_num = filename.replace("{batch_num}", str(batch_number))

                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
                filename_with_batch_num = filename_with_batch_num.replace("{time}", timestamp)
                file_name = f"{filename_with_batch_num}.png"
                img.save(
                    os.path.join(full_output_folder, file_name),
                    pnginfo=metadata,
                    compress_level=self.compress_level,
                )
                results.append({"filename": file_name, "subfolder": subfolder, "type": self.type})
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
            pil = tensor_to_pil(i)
            name = f"{name}.{type}"
            pil.save(os.path.join(directory, name), compress_level=compress_level)

        return ()


class ResizeImageNode:
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
                "divisible_by_2": (IO.BOOLEAN, {"default": False}),
            }
        }

    RETURN_TYPES = (IO.IMAGE, IO.INT, IO.INT)
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "node_function"
    OUTPUT_NODE = True
    CATEGORY = "Fair/image"

    def node_function(self, image, resize_to, side, interpolation, divisible_by_2):
        image = image.movedim(-1, 1)

        image_height, image_width = image.shape[-2:]

        longer_side = "height" if image_height > image_width else "width"
        shorter_side = "height" if image_height < image_width else "width"

        height, width = 0, 0

        if side == "shortest":
            side = shorter_side
        elif side == "longest":
            side = longer_side

        resize_to = float(resize_to)
        if side == "width":
            width = resize_to
            height = image_height * (resize_to / image_width)
        elif side == "height":
            width = image_width * (resize_to / image_height)
            height = resize_to

        width = int(width)
        height = int(height)

        if divisible_by_2:
            width = width - width % 2
            height = height - height % 2

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
        img_out = pil_to_tensor(pil)
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
                pil = rgba_to_rgb(pil)
            elif pil.mode == "RGB" and channels == "RGBA":
                pil = pil.convert("RGBA")

            image_tensor = pil_to_tensor(pil)
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
        out_image = list_to_batch(out_image)

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

        pil = tensor_to_pil(tensor)
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
        image_tensor = pil_to_tensor(new_pil)
        return image_tensor

    def node_function(self, image, alpha_threshold, r, g, b):
        tensor_filled = self.fill_alpha(image, alpha_threshold, (r, g, b))
        return (tensor_filled,)


class FillColorNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IO.IMAGE, {"default": "", "forceInput": True}),
                "color": (IO.COLOR,),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    FUNCTION = "node_function"
    CATEGORY = "Fair/image"

    def fill_color(self, image, color):
        # change white to fill color
        pil = tensor_to_pil(image)

        pixels = pil.getdata()

        new_pixels = []
        for pixel in pixels:
            r, g, b = pixel
            l = int(0.299 * r + 0.587 * g + 0.114 * b)
            # color lerp
            new_pixels.append((int(r + (color[0] - r) * (1 - l / 255.0)), int(g + (color[1] - g) * (1 - l / 255.0)), int(b + (color[2] - b) * (1 - l / 255.0))))

        new_pil = Image.new("RGB", pil.size)
        new_pil.putdata(new_pixels)
        image_tensor = pil_to_tensor(new_pil)
        return image_tensor

    def node_function(self, image, color):
        tensor_filled = self.fill_color(image, color)
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
        pil = tensor_to_pil(image)
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
        image = pil_to_tensor(pil)
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


class ImageShapeNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (IO.IMAGE, {"defaultInput": True}),
            }
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/image"
    RETURN_TYPES = (IO.INT, IO.INT, IO.INT, IO.INT)
    RETURN_NAMES = ("batch", "width", "height", "channel")

    def node_function(self, images):
        # [b,h,w,c]
        batch, height, width, channel = (images.shape[0], images.shape[1], images.shape[2], images.shape[3])
        return (batch, width, height, channel)


class ModulationNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (IO.IMAGE, {"defaultInput": True}),
                "direction": ("ModulationDirection", {"default": "up_to_down"}),
                "speed": (IO.FLOAT, {"default": 0.01, "step": 0.01}),
            }
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/image"
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("images",)

    def node_function(self, images, direction, speed):
        out_images = []

        images_np = []
        for image in images:
            pil = tensor_to_pil(image).convert("L")
            img_np = pil_to_np(pil)
            images_np.append(img_np)

        # Try ProcessPool first with proper error handling
        try:
            images_np = process_modulation_ProcessPool(images_np, direction, speed)
        except Exception as e:
            print(f"ProcessPool failed: {e}, using ThreadPool fallback")
            # Fallback to ThreadPool (already handled in process_modulation_ProcessPool)
            try:
                images_np = process_modulation(images_np, direction, speed)
            except Exception as inner_e:
                print(f"ThreadPool also failed: {inner_e}, processing sequentially")
                # Last resort: sequential processing
                from .modulation import modulation

                processed_images = []
                for idx, image_np in enumerate(images_np):
                    try:
                        processed = modulation(image_np, idx, direction, speed)
                        processed_images.append(processed)
                    except Exception as seq_e:
                        print(f"Error processing image {idx}: {seq_e}, keeping original")
                        processed_images.append(image_np)
                images_np = processed_images

        for image_np in images_np:
            pil = np_to_pil(image_np).convert("RGB")
            image = pil_to_tensor(pil)
            out_images.append(image)

        out_images = torch.stack(out_images, dim=0)
        return (out_images,)


class ModulationDirectionNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "direction": (["up_to_down", "down_to_up", "left_to_right", "right_to_left"], {"default": "up_to_down"}),
            }
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/image"
    RETURN_TYPES = ("ModulationDirection",)
    RETURN_NAMES = ("direction",)

    def node_function(self, direction):
        return (direction,)


class ImageRemoveAlphaNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (IO.IMAGE, {"defaultInput": True}),
                "masks": (IO.MASK, {"defaultInput": True}),
                "fill_color": (IO.STRING, {"default": "#FFFFFF"}),
            }
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/image"
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("images",)

    def node_function(self, images, masks, fill_color):
        out_images = []
        for image, mask in zip(images, masks):
            image_pil = tensor_to_pil(image)
            mask_pil = tensor_to_pil(1 - mask)

            new_pil = Image.new("RGBA", image_pil.size, fill_color)
            new_pil.paste(image_pil, mask_pil)
            new_pil = new_pil.convert("RGB")

            image = pil_to_tensor(new_pil)
            out_images.append(image)

        out_images = torch.stack(out_images, dim=0)
        return (out_images,)


class MaskMapNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metallic": (IO.IMAGE,),
                "ambient_occlusion": (IO.IMAGE,),
                "detail_mask": (IO.IMAGE,),
                "smoothness": (IO.IMAGE,),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/image"
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("mask_map",)

    def node_function(self, metallic, ambient_occlusion, detail_mask, smoothness):
        mask_maps = []
        for metallic_tensor, ambient_occlusion_tensor, detail_mask_tensor, smoothness_tensor in zip(metallic, ambient_occlusion, detail_mask, smoothness):
            if len(metallic_tensor.shape) == 2:
                metallic_pil_single = tensor_to_pil(metallic_tensor)
                metallic_pil = metallic_pil_single.convert("RGB")
            else:
                metallic_pil = tensor_to_pil(metallic_tensor)

            if len(ambient_occlusion_tensor.shape) == 2:
                ambient_occlusion_pil_single = tensor_to_pil(ambient_occlusion_tensor)
                ambient_occlusion_pil = ambient_occlusion_pil_single.convert("RGB")
            else:
                ambient_occlusion_pil = tensor_to_pil(ambient_occlusion_tensor)

            if len(detail_mask_tensor.shape) == 2:
                detail_mask_pil_single = tensor_to_pil(detail_mask_tensor)
                detail_mask_pil = detail_mask_pil_single.convert("RGB")
            else:
                detail_mask_pil = tensor_to_pil(detail_mask_tensor)

            if len(smoothness_tensor.shape) == 2:
                smoothness_pil_single = tensor_to_pil(smoothness_tensor)
                smoothness_pil = smoothness_pil_single.convert("RGB")
            else:
                smoothness_pil = tensor_to_pil(smoothness_tensor)

            metallic_pil_r = metallic_pil.split()[0]
            ambient_pil_occlusion_r = ambient_occlusion_pil.split()[0]
            detail_pil_r = detail_mask_pil.split()[0]
            smoothness_pil_r = smoothness_pil.split()[0]

            mask_map_pil = Image.merge("RGBA", (metallic_pil_r, ambient_pil_occlusion_r, detail_pil_r, smoothness_pil_r))
            mask_map_tensor = pil_to_tensor(mask_map_pil)
            mask_maps.append(mask_map_tensor)
        mask_maps = torch.stack(mask_maps, dim=0)
        return (mask_maps,)


class DetailMapNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "albedo": (IO.IMAGE,),
                "normal": (IO.IMAGE,),
                "smoothness": (IO.IMAGE,),
            }
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/image"
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("detail_map",)

    def node_function(self, albedo, normal, smoothness):
        detail_maps = []
        for albedo_tensor, normal_tensor, smoothness_tensor in zip(albedo, normal, smoothness):
            if len(albedo_tensor.shape) == 2:
                albedo_pil_single = tensor_to_pil(albedo_tensor)
                albedo_pil = albedo_pil_single.convert("RGB")
            else:
                albedo_pil = tensor_to_pil(albedo_tensor)

            if len(normal_tensor.shape) == 2:
                normal_pil_single = tensor_to_pil(normal_tensor)
                normal_pil = normal_pil_single.convert("RGB")
            else:
                normal_pil = tensor_to_pil(normal_tensor)

            if len(smoothness_tensor.shape) == 2:
                smoothness_pil_single = tensor_to_pil(smoothness_tensor)
                smoothness_pil = smoothness_pil_single.convert("RGB")
            else:
                smoothness_pil = tensor_to_pil(smoothness_tensor)

            desaturate_albedo_pil = ImageOps.grayscale(albedo_pil)
            normal_pil_r = normal_pil.split()[0]
            normal_pil_g = normal_pil.split()[1]
            smoothness_pil_r = smoothness_pil.split()[0]

            # R:desaturate_albedo
            # G:normal_pil_g
            # B:smoothness_r
            # A:normal_pil_r
            detail_map_pil = Image.merge("RGBA", (desaturate_albedo_pil, normal_pil_g, smoothness_pil_r, normal_pil_r))
            detail_map = pil_to_tensor(detail_map_pil)
            detail_maps.append(detail_map)
        detail_maps = torch.stack(detail_maps, dim=0)

        return (detail_maps,)


class RoughnessToSmoothnessNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "roughness": (IO.IMAGE,),
            }
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/image"
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("smoothness",)

    def node_function(self, roughness):
        smoothness = 1.0 - roughness
        return (smoothness,)


class PureColorImageNode:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color_hex": (IO.STRING, {"default": "#FFFFFF"}),
                "width": (IO.INT, {"default": 1024, "min": 1, "max": 8192}),
                "height": (IO.INT, {"default": 1024, "min": 1, "max": 8192}),
            }
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/image"
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("pure_color_image",)

    def node_function(self, color_hex, width, height):
        # color is hex format
        color_255 = tuple(int(color_hex.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        pil = Image.new("RGB", (width, height), color=color_255)
        pure_color_image = pil_to_tensor(pil)
        pure_color_image = pure_color_image.unsqueeze(0)
        return (pure_color_image,)


class SaveImageToFolderNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (IO.IMAGE,),
                "folder_path": (IO.STRING, {"default": os.path.join(os.path.expanduser("~"), "Desktop")}),
                "filename_prefix": (IO.STRING, {"default": "ComfyUI_{time}_{batch_num}"}),
            }
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/image"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True

    def node_function(self, images, folder_path, filename_prefix):
        for index, image in enumerate(images):
            if len(image.shape) == 2:
                pil_single = tensor_to_pil(image)
                pil = pil_single.convert("RGB")
            else:
                pil = tensor_to_pil(image)

            filename = filename_prefix
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
            filename = filename.replace("{time}", timestamp)
            filename = filename.replace("{batch_num}", str(index))

            pil.save(os.path.join(folder_path, f"{filename}.png"))
        return ()


# class SaveWEBMNode:
#     def __init__(self):
#         pass

#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "images": (IO.IMAGE,),
#                 "folder_path": (IO.STRING, {"default": os.path.join(os.path.expanduser("~"), "Desktop")}),
#                 "filename_prefix": (IO.STRING, {"default": "ComfyUI_{time}_{batch_num}"}),
#             }
#         }

#     FUNCTION = "node_function"
#     CATEGORY = "Fair/image"
#     RETURN_TYPES = ()
#     RETURN_NAMES = ()
#     OUTPUT_NODE = True

#     def node_function(self, images, folder_path, filename_prefix):
#         for index, image in enumerate(images):
#             if len(image.shape) == 2:
#                 pil_single = tensor_to_pil(image)
#                 pil = pil_single.convert("RGB")
#             else:
#                 pil = tensor_to_pil(image)

#             filename = filename_prefix
#             now = datetime.now()
#             timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
#             filename = filename.replace("{time}", timestamp)
#             filename = filename.replace("{batch_num}", str(index))

#             pil.save(os.path.join(folder_path, f"{filename}.png"))
#         return ()


# https://github.com/theamusing/perfectPixel
class PerfectPixelNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (IO.IMAGE,),
                "sample_method": (["center", "majority"], {"default": "center"}),
                # "grid_size": (IO.INT, {"default": 4}),
                "min_size": (IO.INT, {"default": 4}),
                "peak_width": (IO.INT, {"default": 4}),
                "refine_intensity": (IO.FLOAT, {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.01}),
                "fix_square": (IO.BOOLEAN, {"default": True}),
                "debug": (IO.BOOLEAN, {"default": False}),
            }
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/image"
    RETURN_TYPES = (IO.IMAGE, IO.INT, IO.INT)
    RETURN_NAMES = ("scaled_image", "refined_w", "refined_h")

    def tensor_to_cv2(self, tensor):
        numpy_image = tensor.numpy() * 255.0
        return numpy_image

    def np_to_pil(self, np_image):
        return Image.fromarray(np_image.astype("uint8"), "RGB")

    def node_function(self, images, sample_method, min_size, peak_width, refine_intensity, fix_square, debug):
        outs = []
        for image in images:
            image_cv2 = self.tensor_to_cv2(image)
            w, h, out = get_perfect_pixel(
                image_cv2,
                sample_method=sample_method,
                min_size=min_size,
                peak_width=peak_width,
                refine_intensity=refine_intensity,
                fix_square=fix_square,
                debug=debug,
            )
            out_pil = self.np_to_pil(out)
            image_tensor = pil_to_tensor(out_pil)
            outs.append(image_tensor)
        outs = torch.stack(outs, dim=0)
        return (outs, w, h)
