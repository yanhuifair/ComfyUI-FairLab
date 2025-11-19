import googletrans
from googletrans import Translator
from comfy.comfy_types.node_typing import IO
import asyncio


async def translate_text(string, src, dest):
    async with Translator() as translator:
        translated_text = ""
        if string and string.strip() != "":
            translated_text = await translator.translate(string, src=src, dest=dest)

        return translated_text.text if hasattr(translated_text, "text") else ""


class StringTranslateNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        language_list = list(googletrans.LANGUAGES.keys())
        return {
            "required": {
                "string": (IO.STRING, {"defaultInput": False, "multiline": True}),
                "src": (language_list, {"default": "zh" if "zh" in language_list else "auto"}),
                "dest": (language_list, {"default": "en" if "en" in language_list else "en"}),
            }
        }

    CATEGORY = "Fair/string"
    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "node_function"

    def node_function(self, string, src, dest):
        try:
            # Check if there's already a running event loop
            loop = asyncio.get_running_loop()
            # If we're here, there's a running loop, so we need to use a different approach
            import nest_asyncio

            nest_asyncio.apply()
            text_translated = asyncio.run(translate_text(string, src, dest))
        except RuntimeError:
            # No running loop, safe to use asyncio.run()
            text_translated = asyncio.run(translate_text(string, src, dest))
        return (text_translated,)
