import requests
from comfy.comfy_types.node_typing import IO

# googletrans 4.0.0rc1 is abandoned upstream and pins httpx==0.13; against any modern
# httpx it dies while parsing its own URL ("InvalidURL: Invalid port") before a request
# is ever sent, so the node could never translate. ComfyUI cannot hold httpx at 0.13 for
# this node, so we call the endpoint googletrans itself wrapped — plain GET, no API key,
# no extra dependency beyond requests, which ComfyUI already requires.
TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"
# The text rides in the query string, so keep each request well inside URL length limits.
MAX_CHUNK = 4500

LANGUAGES = {
    "auto": "Detect language",
    "en": "English",
    "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "uk": "Ukrainian",
    "ar": "Arabic",
    "he": "Hebrew",
    "hi": "Hindi",
    "bn": "Bengali",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "tr": "Turkish",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "cs": "Czech",
    "sk": "Slovak",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "bg": "Bulgarian",
    "sr": "Serbian",
    "hr": "Croatian",
    "fa": "Persian",
    "ta": "Tamil",
    "sw": "Swahili",
    "la": "Latin",
    "eo": "Esperanto",
}


def split_chunks(text, limit=MAX_CHUNK):
    # Split only at whitespace inside over-long lines; whole lines move around intact so
    # the joined result keeps its original line structure.
    chunks = []
    current = ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            cut = line.rfind(" ", 0, limit)
            cut = cut + 1 if cut != -1 else limit
            chunks.append(line[:cut])
            line = line[cut:]
        if current and len(current) + len(line) > limit:
            chunks.append(current)
            current = ""
        current += line
    if current:
        chunks.append(current)
    return chunks


def translate_text(string, src, dest):
    params = {"client": "gtx", "dt": "t", "sl": src, "tl": dest, "q": string}
    try:
        response = requests.get(TRANSLATE_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as error:
        raise ValueError(f"translation request failed ({type(error).__name__}): {error}") from error

    # Response is [[["译文","原文",...], ...], null, detected_lang, ...]; a rejected
    # language pair comes back as [null,...] rather than an HTTP error.
    segments = payload[0] if isinstance(payload, list) else None
    if not segments:
        raise ValueError(f"translation endpoint returned no translation for {string[:60]!r} ({src} -> {dest})")
    return "".join(segment[0] for segment in segments if segment and segment[0])


class StringTranslateNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": (IO.STRING, {"defaultInput": False, "multiline": True}),
                "src": (list(LANGUAGES.keys()), {"default": "auto"}),
                "dest": ([code for code in LANGUAGES if code != "auto"], {"default": "en"}),
            }
        }

    CATEGORY = "Fair/string"
    RETURN_TYPES = (IO.STRING,)
    RETURN_NAMES = ("string",)
    FUNCTION = "node_function"
    DESCRIPTION = "Translate text using the Google Translate endpoint (no API key). src 'auto' detects the language."
    SEARCH_ALIASES = ["translate text", "language translate"]

    def node_function(self, string, src, dest):
        if not string or not string.strip():
            return ("",)
        if src == dest:
            return (string,)

        translated = "".join(translate_text(chunk, src, dest) for chunk in split_chunks(string))
        if not translated:
            raise ValueError(f"translation returned empty text for {string[:60]!r} ({src} -> {dest})")
        return (translated,)
