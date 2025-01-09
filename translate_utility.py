from googletrans import Translator, LANGUAGES

translator = Translator()


def translate(prompt, srcTrans=None, toTrans=None):
    if not srcTrans:
        srcTrans = "auto"

    if not toTrans:
        toTrans = "en"

    translate_text_prompt = ""
    if prompt and prompt.strip() != "":
        translate_text_prompt = translator.translate(prompt, src=srcTrans, dest=toTrans)

    return translate_text_prompt.text if hasattr(translate_text_prompt, "text") else ""


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
    FUNCTION = "function"

    def function(self, **kwargs):
        text = kwargs.get("text")
        clip = kwargs.get("clip")

        text_tranlsated = translate(text)
        tokens = clip.tokenize(text_tranlsated)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], text_tranlsated)


class TranslateStringNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sting": ("STRING", {"multiline": True}),
            }
        }

    CATEGORY = "Fair/string"

    RETURN_TYPES = ("STRING",)
    FUNCTION = "function"

    def function(self, sting):
        text_tranlsated = translate(sting)
        return (text_tranlsated,)
