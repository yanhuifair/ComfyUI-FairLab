# https://github.com/ollama/ollama-python
# https://github.com/ollama/ollama/blob/main/docs/api.md#version
import ollama
from ollama import Client

from .image_utility import tensor2pil
from .image_utility import pil_to_base64
import re


class OllamaNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # cls.model_list = [model.model for model in ollama.list().models]
        return {
            "required": {
                "prompt": ("STRING", {"default": "describe the image", "multiline": True, "tooltip": "the prompt to generate a response for"}),
                "url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "model": ((), {"tooltip": "(required) the model name"}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "(optional) a list of base64-encoded images"}),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/ollama"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("think", "response")

    def separate_response(self, response):
        pattern = r"<think>\n(.*?)\n</think>\s*(.*)"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            think_content = match.group(1).strip()
            response = match.group(2).strip()
        else:
            think_content = ""
            response = response.strip()

        return (think_content, response)

    def node_function(self, prompt, url, model, debug, images=None):
        out_think = ""
        out_response = ""

        base64_images = []
        if images is not None:
            for image in images:
                pil = tensor2pil(image)
                base64_images.append(pil_to_base64(pil))

        client = Client(host=url)
        try:
            ollama_response = client.generate(
                model=model,
                prompt=prompt,
                images=base64_images,
            )
        except ollama.ResponseError as e:
            print(f"Ollama Error: {e.error}")
            return ("", "")

        if debug:
            print(f"🦙 Ollama debug:\n{ollama_response}")

        out_think, out_response = self.separate_response(ollama_response.response)
        print(f"🦙 Ollama think:\n{out_think}")
        print(f"🦙 Ollama response:\n{out_response}")
        return (out_think, out_response)
