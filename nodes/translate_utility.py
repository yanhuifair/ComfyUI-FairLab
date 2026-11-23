import asyncio

import googletrans
from comfy.comfy_types.node_typing import IO
from googletrans import Translator


async def translate_text(string, src, dest):
    async with Translator() as translator:
        return await translator.translate(string, src=src, dest=dest)


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
    RETURN_NAMES = ("string",)
    FUNCTION = "node_function"
    DESCRIPTION = "Translate text between supported languages using googletrans."
    SEARCH_ALIASES = ["translate text", "language translate"]

    def node_function(self, string, src, dest):
        if not string or not string.strip():
            return ("",)

        try:
            translated = self._translate_once(string, src, dest)
        except Exception as error:
            # Surface the cause: the previous version returned "", so a dead googletrans
            # or a network error looked like "the text translated to nothing".
            raise ValueError(f"translation failed ({type(error).__name__}): {error}") from error

        text = getattr(translated, "text", None)
        if not text:
            raise ValueError(f"googletrans returned no translation for {string!r} ({src} -> {dest})")
        return (text,)

    def _translate_once(self, string, src, dest):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop in this thread, so asyncio.run() is safe as is.
            return asyncio.run(translate_text(string, src, dest))

        # ComfyUI executes node functions inside its own event loop, so asyncio.run() only
        # works after the re-entrant patch; before this the RuntimeError from a nested
        # asyncio.run() was caught below and retried, then raised a second time.
        import nest_asyncio

        nest_asyncio.apply()
        return asyncio.run(translate_text(string, src, dest))
