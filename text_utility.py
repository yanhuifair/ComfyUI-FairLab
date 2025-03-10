from comfy.utils import ProgressBar
import os
import string


class SaveStringToDirectoryNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"defaultInput": True}),
                "directory": ("STRING", {"defaultInput": True}),
                "name": ("STRING", {"defaultInput": True}),
            }
        }

    RETURN_TYPES = ()
    CATEGORY = "Fair/string"

    FUNCTION = "function"
    OUTPUT_NODE = True

    def function(self, string, directory, name):
        for file_string, file_directory, file_name in zip(string, directory, name):
            file_name_with_suffix = f"{file_name}.txt"
            full_path = os.path.join(file_directory, file_name_with_suffix)
            with open(full_path, "w", encoding="utf-8") as file:
                file.write(file_string)

        return ()


class FixUTF8StringNode:
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
        result = "".join(replacements.get(c, c) if c not in printable else c for c in text)
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


class StringCombineNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"defaultInput": True}),
                "combine": ("STRING", {"defaultInput": True}),
                "combine_at": (
                    ["start", "end"],
                    {
                        "default": "start",
                        "defaultInput": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "Fair/string"

    FUNCTION = "function"
    OUTPUT_NODE = True

    def function(self, string, combine, combine_at):
        if type(string).__name__ == "list":
            out_string_list = []

            for sc in string:
                if combine_at == "start":
                    combined = combine + sc
                else:
                    combined = sc + combine
                out_string_list.append(combined)

            return (out_string_list,)

        else:

            if combine_at == "start":
                combined = combine + string
            else:
                combined = string + combine
            return (combined,)


class StringFieldNode:
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


class SequenceStringListNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {}

    RETURN_TYPES = ("STRING",)
    CATEGORY = "Fair/string"

    FUNCTION = "function"
    OUTPUT_NODE = True

    def function(self):
        out = []
        for i in range(10):
            out.append(f"{i}")
        return (out,)


class PrependTagsNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"defaultInput": True}),
                "tags": ("STRING", {"defaultInput": False, "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "Fair/string"

    FUNCTION = "function"
    OUTPUT_NODE = True

    def function(self, string, tags):
        out_string_list = []
        for st in string:
            input_tags_list = [s.strip() for s in st.split(",")]
            prepend_tags_list = [s.strip() for s in tags.split(",")]
            out_string = prepend_tags_list + input_tags_list
            out_string = ", ".join(out_string)
            out_string_list.append(out_string)
        return (out_string_list,)


class AppendTagsNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"defaultInput": True}),
                "tags": ("STRING", {"defaultInput": False, "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "Fair/string"

    FUNCTION = "function"
    OUTPUT_NODE = True

    def function(self, string, tags):
        out_string_list = []
        for st in string:
            input_tags_list = [s.strip() for s in st.split(",")]
            append_tags_list = [s.strip() for s in tags.split(",")]
            out_string = input_tags_list + append_tags_list
            out_string = ", ".join(out_string)
            out_string_list.append(out_string)
        return (out_string_list,)


class BlacklistTagsNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"defaultInput": True}),
                "tags": ("STRING", {"defaultInput": False, "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "Fair/string"

    FUNCTION = "function"
    OUTPUT_NODE = True

    def function(self, string, tags):
        out_string_list = []
        for st in string:
            input_tags_list = [s.strip() for s in st.split(",")]
            blacklist_tags_list = [s.strip() for s in tags.split(",")]
            b_set = set(blacklist_tags_list)
            out_string = [x for x in input_tags_list if x not in b_set]
            out_string = ", ".join(out_string)
            out_string_list.append(out_string)
        return (out_string_list,)
