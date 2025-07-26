import ollama
from ollama import Client

from .image_utility import tensor2pil
from .image_utility import pil_to_base64


class OllamaVisionNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # cls.model_list = [model.model for model in ollama.list().models]
        return {
            "required": {
                "prompt": ("STRING", {"default": "describe the image", "multiline": True}),
                "image": ("IMAGE",),
                "url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "model": ((), {}),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/ollama"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)

    def node_function(self, prompt, image, url, model):
        response_out = ""
        img = image[0]
        pil = tensor2pil(img)
        base64_image = pil_to_base64(pil)

        client = Client(host=url)

        response = client.generate(
            model=model,
            prompt=prompt,
            images=[base64_image],
        )
        response_out = response.response
        print(f"Ollama response:\n{response_out}")
        return (response_out,)
