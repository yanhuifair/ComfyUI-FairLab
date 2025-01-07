import os
import json
import numpy as np

from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo

import folder_paths
import comfy.utils
from comfy.cli_args import args
from datetime import datetime
from comfy.utils import ProgressBar

import torch.nn.functional as NNF


class DownloadImageClass:
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
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
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
            filename_with_batch_num = filename_with_batch_num.replace(
                "%time%", timename
            )
            file = f"{filename_with_batch_num}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return {"ui": {"images": results}}


class SaveImagesToFolderClass:
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


class SaveImageToFolderClass:
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


class ImageResizeClass:
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

        image = comfy.utils.common_upscale(
            image, new_width, new_height, interpolation, "center"
        )

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
