import folder_paths
import comfy.sd
import comfy.utils
from comfy.comfy_types.node_typing import IO
from nodes import LoraLoader


class PrintAnyNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (IO.ANY,),
                "log": (IO.STRING,),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/utility"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    def node_function(self, input, log):
        print(f"log: {log}")
        print(f"type: {type(input)}")
        print(input)

        return ()


class PrintImageNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IO.IMAGE,),
                "log": (IO.STRING,),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/utility"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    def node_function(self, image, log):
        print(f"log: {log}")
        print(f"shape: {image.shape}")
        print(image)
        return ()


class LoraLoaderDualNode:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (IO.MODEL, {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength_model": (IO.FLOAT, {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": (IO.FLOAT, {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
                "lora_name_2": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength_model_2": (IO.FLOAT, {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip_2": (IO.FLOAT, {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            }
        }

    RETURN_TYPES = (IO.MODEL, IO.CLIP)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "node_function"

    CATEGORY = "Fair/loaders"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."

    OUTPUT_IS_LIST = (True, True)


class LoraLoaderDualNode:
    def __init__(self):
        self.loraLoader = LoraLoader()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (IO.MODEL, {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength_model": (IO.FLOAT, {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": (IO.FLOAT, {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
                "lora_name_2": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength_model_2": (IO.FLOAT, {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip_2": (IO.FLOAT, {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            }
        }

    RETURN_TYPES = (IO.MODEL, IO.CLIP)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "node_function"

    CATEGORY = "loaders"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."

    OUTPUT_IS_LIST = (True, True)

    def node_function(self, model, clip, lora_name, strength_model, strength_clip, lora_name_2, strength_model_2, strength_clip_2):
        model_lora, clip_lora = self.loraLoader.load_lora(model, clip, lora_name, strength_model, strength_clip)
        model_lora_2, clip_lora_2 = self.loraLoader.load_lora(model, clip, lora_name_2, strength_model_2, strength_clip_2)
        model_loras = [model_lora, model_lora_2]
        clip_loras = [clip_lora, clip_lora_2]
        return (model_loras, clip_loras)
