from comfy.utils import ProgressBar
import os
import string


class SaveStringToFolderClass:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"defaultInput": True}),
                "folder": ("STRING", {"defaultInput": True}),
                "name": ("STRING", {"defaultInput": True}),
            }
        }

    RETURN_TYPES = ()
    CATEGORY = "Fair/string"

    FUNCTION = "function"
    OUTPUT_NODE = True

    def function(self, string, folder, name):
        for content, file_name in zip(string, name):
            file_name_suffix = f"{file_name}.txt"
            full_path = os.path.join(folder, file_name_suffix)
            with open(full_path, "w", encoding="utf-8") as file:
                file.write(content)

        return ()


class FixUTF8StringClass:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"string": ("STRING", {"defaultInput": True})}}

    RETURN_TYPES = ("STRING",)
    CATEGORY = "Fair/string"

    FUNCTION = "function"
    OUTPUT_NODE = True

    # Function to replace special characters with basic equivalents
    def replace_special_characters(self, text):
        replacements = {
            "（": "(",
            "）": ")",
            "，": ",",
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "“": '"',
            "”": '"',
            "–": "-",
            "—": "-",
            "…": "...",
            "é": "e",
            "è": "e",
            "ê": "e",
            "á": "a",
            "à": "a",
            "â": "a",
            "ó": "o",
            "ò": "o",
            "ô": "o",
            "ú": "u",
            "ù": "u",
            "û": "u",
            "í": "i",
            "ì": "i",
            "î": "i",
            "ç": "c",
            "ñ": "n",
            "ß": "ss",
            "ü": "u",
            "ö": "o",
            "ä": "a",
            "ø": "o",
            "æ": "ae",
            # Add more replacements as needed
        }

        # Remove all characters not in printable set or replace if in the replacements dictionary
        printable = set(string.printable)
        result = "".join(
            replacements.get(c, c) if c not in printable else c for c in text
        )
        return result

    def remove_special_characters(self, text):
        # Keep only ASCII characters (characters with ordinal values from 0 to 127)
        return "".join(c if ord(c) < 128 else "" for c in text)

    def function(self, string):
        # Replace special characters
        out_string = []
        for content in string:
            cleaned_content = self.replace_special_characters(content)
            cleaned_content = self.remove_special_characters(cleaned_content)
            out_string.append(cleaned_content)
        return (out_string,)


class StringCombineClass:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"defaultInput": True}),
                "string_combine": ("STRING", {"defaultInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "Fair/string"

    FUNCTION = "function"
    OUTPUT_NODE = True

    def function(self, string, string_combine):
        out_string_list = []
        for string_combine in string_combine:
            combine_string = string + string_combine
            out_string_list.append(combine_string)
        return (out_string_list,)


class StringFieldClass:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"string": ("STRING", {"multiline": True})}}

    RETURN_TYPES = ("STRING",)
    CATEGORY = "Fair/string"

    FUNCTION = "function"
    OUTPUT_NODE = True

    def function(self, string):
        return (string,)
