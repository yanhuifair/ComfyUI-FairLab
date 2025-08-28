# https://github.com/ollama/ollama-python
# https://github.com/ollama/ollama/blob/main/docs/api.md#version
import ollama
from ollama import Client

from .image_utility import tensor2pil
from .image_utility import pil_to_base64
import re
from comfy.comfy_types.node_typing import IO


class OllamaClientNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": (IO.STRING, {"default": "http://127.0.0.1:11434"}),
                "model": ((), {}),
                "keep_alive": (IO.INT, {"default": 5, "tooltip": "controls how long the model will stay loaded into memory following the request, The unit is minutes.\nSet to -1 to keep the model loaded indefinitely.", "min": -1}),
            }
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/ollama"
    OUTPUT_NODE = True
    RETURN_TYPES = ("ollama_connection",)
    RETURN_NAMES = ("ollama_connection",)

    def node_function(self, url, model, keep_alive):
        ollama_client = Client(host=url)
        ollama_connection = (ollama_client, model, keep_alive)
        return (ollama_connection,)


class OllamaNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # cls.model_list = [model.model for model in ollama.list().models]
        return {
            "required": {
                "prompt": (IO.STRING, {"default": "describe the image", "multiline": True, "tooltip": "the prompt to generate a response for"}),
                "ollama_connection": ("ollama_connection", {"forceInput": True}),
            },
            "optional": {
                "images": (IO.IMAGE, {"tooltip": "(optional) a list of images"}),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/ollama"
    OUTPUT_NODE = True
    RETURN_TYPES = (IO.STRING, IO.STRING)
    RETURN_NAMES = ("think", "response")

    def separate_response(self, response):

        out_think = ""
        out_response = ""

        pattern = r"<think>\n(.*?)\n</think>\s*(.*)"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            out_think = match.group(1).strip()
            out_response = match.group(2).strip()
        else:
            out_response = response.strip()

        return (out_think, out_response)

    def node_function(self, prompt, ollama_connection, images=None):
        ollama_client = ollama_connection[0]
        ollama_model = ollama_connection[1]
        keep_alive = ollama_connection[2]

        out_think = ""
        out_response = ""

        base64_images = []
        if images is not None:
            for image in images:
                pil = tensor2pil(image)
                base64_images.append(pil_to_base64(pil))

        try:
            ollama_response = ollama_client.generate(
                model=ollama_model,
                prompt=prompt,
                images=base64_images,
                keep_alive=keep_alive,
            )
        except ollama.ResponseError as e:
            print(f"Ollama Error: {e.error}")
            return ("", "")

        out_think, out_response = self.separate_response(ollama_response.response)
        if out_think != "":
            print(f"🦙 Ollama think:\n{out_think}")
        if out_response != "":
            print(f"🦙 Ollama response:\n{out_response}")
        return (out_think, out_response)
