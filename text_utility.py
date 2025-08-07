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
                "extension": ([".txt", ".cap", ".caption"], {"defaultInput": False, "default": ".txt"}),
            }
        }

    CATEGORY = "Fair/string"
    FUNCTION = "node_function"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("string", "directory", "name")

    def node_function(self, string, directory, name, extension):
        file_name_with_suffix = f"{name}{extension}"
        full_path = os.path.join(directory, file_name_with_suffix)
        with open(full_path, "w", encoding="utf-8") as file:
            file.write(string)

        return (string, directory, name)


class LoadStringFromDirectoryNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"defaultInput": True}),
            }
        }

    CATEGORY = "Fair/string"
    FUNCTION = "node_function"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("string", "directory", "name")
    OUTPUT_IS_LIST = (True, True, True)

    def node_function(self, directory):

        string_list = []
        directory_list = []
        name_without_ext_list = []
        for file_name in os.listdir(directory):
            if file_name.endswith(".txt"):
                with open(os.path.join(directory, file_name), "r", encoding="utf-8") as file:
                    string_list.append(file.read())
                    directory_list.append(directory)
                    name_without_ext_list.append(os.path.splitext(file_name)[0])
        return (string_list, directory_list, name_without_ext_list)


class FixUTF8StringNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"string": ("STRING", {"defaultInput": True})}}

    RETURN_TYPES = ("STRING",)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
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

    def node_function(self, string):
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
                    ["front", "back"],
                    {
                        "default": "back",
                        "defaultInput": False,
                    },
                ),
            }
        }

    CATEGORY = "Fair/string"
    FUNCTION = "node_function"
    RETURN_TYPES = ("STRING",)

    def node_function(self, string, combine, combine_at):
        if combine_at == "front":
            combined = combine + string
        else:
            combined = string + combine
        return (combined,)


class StringNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"string": ("STRING", {"multiline": True, "defaultInput": False})}}

    RETURN_TYPES = ("STRING",)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
    OUTPUT_NODE = True

    def node_function(self, string):
        return (string,)


class IntNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"number": ("INT", {"multiline": False, "defaultInput": False, "default": 0, "max": 65535, "min": -65535})}}

    RETURN_TYPES = ("INT",)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
    OUTPUT_NODE = True

    def node_function(self, number):
        return (number,)


class FloatNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"number": ("FLOAT", {"multiline": False, "defaultInput": False})}}

    RETURN_TYPES = ("FLOAT",)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
    OUTPUT_NODE = True

    def node_function(self, number):
        return (number,)


class SequenceStringListNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {}

    RETURN_TYPES = ("STRING",)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
    OUTPUT_NODE = True

    def node_function(self):
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

    FUNCTION = "node_function"
    OUTPUT_NODE = True

    def node_function(self, string, tags):
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

    FUNCTION = "node_function"
    OUTPUT_NODE = True

    def node_function(self, string, tags):
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

    FUNCTION = "node_function"
    OUTPUT_NODE = True

    def node_function(self, string, tags):
        out_string_list = []
        for st in string:
            input_tags_list = [s.strip() for s in st.split(",")]
            blacklist_tags_list = [s.strip() for s in tags.split(",")]
            b_set = set(blacklist_tags_list)
            out_string = [x for x in input_tags_list if x not in b_set]
            out_string = ", ".join(out_string)
            out_string_list.append(out_string)
        return (out_string_list,)


class ShowStringNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "Fair/string"
    FUNCTION = "node_function"
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True

    def node_function(self, string):
        return {"ui": {"text": string}, "result": (string,)}


class LoadStringNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"forceInput": True}),
            }
        }

    CATEGORY = "Fair/string"
    FUNCTION = "node_function"
    RETURN_TYPES = ("STRING", "STRING", "STRING")

    def node_function(self, path):
        out_string = ""
        out_path = path
        out_name = os.path.basename(path)

        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as file:
                    out_string = file.read()
            else:
                out_string = f"File not found: {path}"
        except Exception as e:
            print(f"Error reading file {path}: {e}")

        return (out_string, out_path, out_name)


class RemoveDuplicateTagsNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"forceInput": True}),
            }
        }

    CATEGORY = "Fair/string"
    FUNCTION = "node_function"
    RETURN_TYPES = ("STRING",)

    def node_function(self, string):
        out_string = ""
        tags = [tag.strip() for tag in string.split(",")]
        unique_tags = set(tags)
        out_string = ", ".join(unique_tags)
        return (out_string,)
