import os
import string
from comfy.comfy_types.node_typing import IO


def string_to_tags(string):
    tags = [s.strip() for s in string.split(",")]
    return tags


def tags_to_string(tags):
    return ", ".join(tags)


class SaveStringToDirectoryNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": (IO.STRING, {"defaultInput": True}),
                "directory": (IO.STRING, {"defaultInput": True}),
                "name": (IO.STRING, {"defaultInput": True}),
                "extension": ([".txt", ".cap", ".caption"], {"defaultInput": False, "default": ".txt"}),
            }
        }

    CATEGORY = "Fair/string"
    FUNCTION = "node_function"
    OUTPUT_NODE = True
    RETURN_TYPES = (IO.STRING, IO.STRING, IO.STRING)
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
                "directory": (IO.STRING, {"defaultInput": True}),
            }
        }

    CATEGORY = "Fair/string"
    FUNCTION = "node_function"
    OUTPUT_NODE = True
    RETURN_TYPES = (IO.STRING, IO.STRING, IO.STRING)
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
        return {"required": {"string": (IO.STRING, {"defaultInput": True})}}

    RETURN_TYPES = (IO.STRING,)
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


class StringAppendNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "front": (IO.STRING,),
                "back": (IO.STRING,),
            }
        }

    CATEGORY = "Fair/string"
    FUNCTION = "node_function"
    RETURN_TYPES = (IO.STRING,)

    def node_function(self, front, back):
        append = front + back
        return (append,)


class StringNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"string": (IO.STRING, {"multiline": True, "defaultInput": False})}}

    RETURN_TYPES = (IO.STRING,)
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
        return {
            "required": {"value": (IO.INT, {"multiline": False, "defaultInput": False, "default": 0, "max": 65535, "min": -65535})},
        }

    RETURN_TYPES = (IO.INT,)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
    OUTPUT_NODE = True

    def node_function(self, value):
        return (value,)


class FloatNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("FLOAT", {"multiline": False, "defaultInput": False})}}

    RETURN_TYPES = ("FLOAT",)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
    OUTPUT_NODE = True

    def node_function(self, value):
        return (value,)


class RangeStringNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": (IO.INT, {"default": 0}),
                "stop": (IO.INT, {"default": 0}),
                "step": (IO.INT, {"default": 1}),
            },
        }

    RETURN_TYPES = (IO.STRING,)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = True

    def node_function(self, start, stop, step):
        number_list = []
        for i in range(start, stop, step):
            number_list.append(f"{i}")
        return (number_list,)


class PrependTagsNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": (IO.STRING, {"defaultInput": True}),
                "tags": (IO.STRING, {"defaultInput": False, "multiline": True}),
            }
        }

    RETURN_TYPES = (IO.STRING,)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
    OUTPUT_NODE = True

    def node_function(self, string, tags):
        input_tags = string_to_tags(string)
        prepend_tags = string_to_tags(tags)
        out_string = prepend_tags + input_tags
        out_string = tags_to_string(out_string)
        return (out_string,)


class AppendTagsNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "front": (IO.STRING, {}),
                "back": (IO.STRING, {}),
            }
        }

    RETURN_TYPES = (IO.STRING,)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
    OUTPUT_NODE = True

    def node_function(self, front, back):
        front_tags = string_to_tags(front)
        back_tags = string_to_tags(back)
        out_tags = front_tags + back_tags
        out_tags = set(out_tags)
        out_string = tags_to_string(out_tags)
        return (out_string,)


class ExcludeTagsNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": (IO.STRING, {"defaultInput": True}),
                "tags": (IO.STRING, {"defaultInput": False, "multiline": True}),
            }
        }

    RETURN_TYPES = (IO.STRING,)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
    OUTPUT_NODE = True

    def node_function(self, string, tags):
        input_tags = string_to_tags(string)
        exclude_tags = string_to_tags(tags)
        exclude_tags = set(exclude_tags)
        out_string = [x for x in input_tags if x not in exclude_tags]
        out_string = tags_to_string(out_string)
        return (out_string,)


class ShowStringNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": (IO.STRING, {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "Fair/string"
    FUNCTION = "node_function"
    RETURN_TYPES = (IO.STRING,)
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
                "path": (IO.STRING, {}),
            }
        }

    CATEGORY = "Fair/string"
    FUNCTION = "node_function"
    RETURN_TYPES = (IO.STRING, IO.STRING, IO.STRING)
    RETURN_NAMES = ("string", "path", "name")

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


class UniqueTagsNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": (IO.STRING, {"forceInput": True}),
            }
        }

    CATEGORY = "Fair/string"
    FUNCTION = "node_function"
    RETURN_TYPES = (IO.STRING,)

    def node_function(self, string):
        out_string = ""
        tags = string_to_tags(string)
        unique_tags = set(tags)
        out_string = tags_to_string(unique_tags)
        return (out_string,)
