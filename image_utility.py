import os
import requests
import json
import numpy as np
import cv2
from io import BytesIO
from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo

import folder_paths
import comfy.utils
from comfy.cli_args import args
from datetime import datetime
from comfy.utils import ProgressBar

import torch
import torch.nn.functional as NNF


def pil2tensor(img):
    output_images = []
    output_masks = []
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        if i.mode == "I":
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
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


def load_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    file_name = url.split("/")[-1]
    return img, file_name


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
                "images": ("IMAGE",),
                "filename_prefix": (
                    "STRING",
                    {"default": "ComfyUI_%time%_%batch_num%"},
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "function"
    OUTPUT_NODE = True
    CATEGORY = "Fair/image"

    def function(
        self,
        images,
        filename_prefix="ComfyUI",
        prompt=None,
        extra_pnginfo=None,
    ):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
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


class SaveImagesToFolderNode:
    def __init__(self):
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"defaultInput": True}),
                "folder": ("STRING", {"defaultInput": True}),
                "names": ("STRING", {"defaultInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "function"
    OUTPUT_NODE = True
    CATEGORY = "Fair/image"

    def function(self, images, folder, names):
        total_bar = len(images)
        pbar = ProgressBar(total_bar)
        processed_bar = 0
        for image, file_name in zip(images, names):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            file = f"{file_name}.png"
            img.save(
                os.path.join(folder, file),
                compress_level=self.compress_level,
            )

            processed_bar += 1
            pbar.update_absolute(processed_bar, total_bar)

        return ()


class SaveImageToFolderNode:
    def __init__(self):
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"defaultInput": True}),
                "folder": ("STRING", {"defaultInput": False}),
                "name": ("STRING", {"defaultInput": False}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "function"
    OUTPUT_NODE = True
    CATEGORY = "Fair/image"

    def function(self, image, folder, name):
        i = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        file = f"{name}.png"
        img.save(
            os.path.join(folder, file),
            compress_level=self.compress_level,
        )

        return ()


class ImageResizeNode:
    def __init__(self):
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resize_to": ("INT", {"default": 1024}),
                "side": (
                    ["shortest", "longest", "width", "height"],
                    {"default": "longest"},
                ),
                "interpolation": (
                    [
                        "lanczos",
                        "nearest",
                        "bilinear",
                        "bicubic",
                        "area",
                        "nearest-exact",
                    ],
                ),
            },
            "optional": {
                "mask_opt": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")

    FUNCTION = "function"
    OUTPUT_NODE = True
    CATEGORY = "Fair/image"

    def function(self, image, resize_to, side, interpolation, mask_opt=None):

        image = image.movedim(-1, 1)

        image_height, image_width = image.shape[-2:]

        longer_side = "height" if image_height > image_width else "width"
        shorter_side = "height" if image_height < image_width else "width"

        new_height, new_width, scale_ratio = 0, 0, 0

        if side == "shortest":
            side = shorter_side
        elif side == "longest":
            side = longer_side

        if side == "width":
            scale_ratio = resize_to / image_width
        elif side == "height":
            scale_ratio = resize_to / image_height

        new_height = image_height * scale_ratio
        new_width = image_width * scale_ratio

        new_width = int(new_width)
        new_height = int(new_height)

        image = comfy.utils.common_upscale(image, new_width, new_height, interpolation, "center")

        if mask_opt is not None:
            mask_opt = mask_opt.permute(0, 1, 2)

            mask_opt = mask_opt.unsqueeze(0)
            mask_opt = NNF.interpolate(
                mask_opt,
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
            )

            mask_opt = mask_opt.squeeze(0)
            mask_opt = mask_opt.squeeze(0)

            mask_opt = mask_opt.permute(0, 1)

        image = image.movedim(1, -1)

        return (image, mask_opt)


class VideoToImagesNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_file": ("STRING", {"defaultInput": False}),
                "capture_rate": ("INT", {"default": 30}),
                "frame_offset": ("INT", {"default": 0}),
                "images_dir": ("STRING", {"defaultInput": False}),
                "images_name_prefix": ("STRING", {"defaultInput": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image_path_list",)
    FUNCTION = "function"
    OUTPUT_NODE = True
    CATEGORY = "Fair/image"

    def function(self, video_file, capture_rate, frame_offset, images_dir, images_name_prefix):
        video_file = video_file.replace('"', "")
        images_dir = images_dir.replace('"', "")

        vc = cv2.VideoCapture(video_file)
        width = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = vc.get(cv2.CAP_PROP_FPS)
        num_frames = vc.get(cv2.CAP_PROP_FRAME_COUNT)
        print("width:{} \nheight:{} \nfps:{} \nnum_frames:{}".format(width, height, fps, num_frames))

        bar_total = num_frames
        progress_bar = ProgressBar(bar_total)

        counter = 1 - frame_offset
        image_path_list = []
        if vc.isOpened():
            while True:
                frame_success, frame_image = vc.read()
                if frame_success:
                    if counter % capture_rate == 0:
                        file_name = str(counter) + ".jpg"
                        file_name = images_name_prefix + file_name
                        image_path = os.path.join(images_dir, file_name)
                        image_path_list.append(image_path)
                        cv2.imwrite(image_path, frame_image)
                else:
                    break
                counter += 1
                progress_bar.update_absolute(counter, bar_total)
        vc.release()

        return (image_path_list,)


class ImagesToVideoNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_dir": ("STRING", {"defaultInput": False}),
                "frame_rate": ("INT", {"default": 30}),
                "video_dir": ("STRING", {"defaultInput": False}),
                "video_name": ("STRING", {"defaultInput": False, "default": "output"}),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = ("image_path_list", "video_path")
    FUNCTION = "function"
    OUTPUT_NODE = True
    CATEGORY = "Fair/image"

    def function(self, images_dir, frame_rate, video_dir, video_name):
        images_dir = images_dir.replace('"', "")
        video_dir = video_dir.replace('"', "")
        video_path = os.path.join(video_dir, f"{video_name}.mp4")

        image_path_list = []
        image_name_list = os.listdir(images_dir)
        for image_path in image_name_list:
            image_path_list.append(os.path.join(images_dir, image_path))

        bar_total = len(image_name_list)
        progress_bar = ProgressBar(bar_total)

        img0 = cv2.imread(image_path_list[0])
        height, width, layers = img0.shape
        print(f"video resolution:{width}*{height}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

        bar_counter = 0
        for image_path in image_path_list:
            img = cv2.imread(image_path)
            video_writer.write(img)

            bar_counter += 1
            progress_bar.update_absolute(bar_counter, bar_total)

        video_writer.release()

        return (
            image_path_list,
            video_path,
        )


class LoadImageFormURLNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"url": ("STRING", {"defaultInput": False, "multiline": True})}}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "file_name")
    FUNCTION = "node_function"
    CATEGORY = "Fair/image"

    def node_function(self, url):
        img, file_name = load_image(url)
        img_out, mask_out = pil2tensor(img)
        return (img_out, mask_out, file_name)


def load_image_to_tensor(folder_path, recursive, convert_to_rgb):
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

    return (image_tensors, image_folders, image_names)


class LoadImageFromDirectoryNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "forceInput": False}),
                "recursive": ("BOOLEAN", {"default": False}),
                "convert_to_rgb": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "directory", "name")
    FUNCTION = "node_function"
    CATEGORY = "Fair/image"

    def node_function(self, directory, recursive, convert_to_rgb):
        if not directory or not os.path.isdir(directory):
            raise Exception("folder_path is not valid: " + directory)

        out_images, out_folders, out_image_names = load_image_to_tensor(directory, recursive, convert_to_rgb)

        return (out_images, out_folders, out_image_names)
