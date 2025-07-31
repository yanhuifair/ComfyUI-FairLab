import asyncio
from googletrans import Translator

translator = Translator()


def translate_text(string, srcTrans=None, toTrans=None):
    if not srcTrans:
        srcTrans = "auto"

    if not toTrans:
        toTrans = "en"

    translated_text = ""
    if string and string.strip() != "":
        translated_text = translator.translate(string, src=srcTrans, dest=toTrans)

    return translated_text.text if hasattr(translated_text, "text") else ""


class CLIPTranslatedNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "placeholder": "Input prompt"}),
            }
        }

    CATEGORY = "Fair/conditioning"

    RETURN_TYPES = ("CONDITIONING", "STRING")
    FUNCTION = "node_function"

    def node_function(self, **kwargs):
        text = kwargs.get("text")
        clip = kwargs.get("clip")

        text_translated = translate_text(text)
        tokens = clip.tokenize(text_translated)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], text_translated)


class TranslateStringNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"defaultInput": True}),
                "translate_mode": (["en_to_cn", "cn_to_en", "auto"], {"default": "en_to_cn"}),
            }
        }

    CATEGORY = "Fair/string"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "node_function"
    OUTPUT_NODE = True

    def node_function(self, string, translate_mode):
        if translate_mode == "en_to_cn":
            text_translated = translate_text(string, "en", "zh-cn")
        elif translate_mode == "cn_to_en":
            text_translated = translate_text(string, "zh-cn", "en")
        elif translate_mode == "auto":
            text_translated = translate_text(string)
        else:
            text_translated = ""
        if text_translated is not None:
            print(f"Translate:\n{text_translated}")
        return (text_translated,)
