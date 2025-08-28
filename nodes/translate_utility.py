import googletrans
from googletrans import Translator
from comfy.comfy_types.node_typing import IO

translator = Translator()


def translate_text(string, src, dest):
    translated_text = ""
    if string and string.strip() != "":
        translated_text = translator.translate(string, src=src, dest=dest)

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
                "src": (language_list, {"default": "en" if "en" in language_list else "auto"}),
                "dest": (language_list, {"default": "zh-cn" if "zh-cn" in language_list else "en"}),
            }
        }

    CATEGORY = "Fair/string"
    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "node_function"

    def node_function(self, string, src, dest):
        text_translated = translate_text(string, src, dest)
        return (text_translated,)
