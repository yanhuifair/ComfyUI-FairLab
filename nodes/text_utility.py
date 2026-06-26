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
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    DESCRIPTION = "Save a string to a file in the target directory."
    SEARCH_ALIASES = ["save text", "write string"]

    def node_function(self, string, directory, name, extension):
        file_name_with_suffix = f"{name}{extension}"
        full_path = os.path.join(directory, file_name_with_suffix)
        with open(full_path, "w", encoding="utf-8") as file:
            file.write(string)

        return ()


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
    DESCRIPTION = "Load all text files from a directory and return their contents with paths."
    SEARCH_ALIASES = ["load texts", "read folder text"]

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
    RETURN_NAMES = ("string",)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
    OUTPUT_NODE = True
    DESCRIPTION = "Normalize common UTF-8 punctuation and strip non-ASCII characters."
    SEARCH_ALIASES = ["clean text", "ascii sanitize"]

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
    RETURN_NAMES = ("string",)
    DESCRIPTION = "Concatenate two strings in order."
    SEARCH_ALIASES = ["concat string", "join text"]

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
    RETURN_NAMES = ("string",)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
    OUTPUT_NODE = True
    DESCRIPTION = "Output a string value from a text widget."
    SEARCH_ALIASES = ["text input", "string input"]

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
    RETURN_NAMES = ("value",)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
    OUTPUT_NODE = True
    DESCRIPTION = "Output an integer value from a numeric widget."
    SEARCH_ALIASES = ["integer input", "int value"]

    def node_function(self, value):
        return (value,)


class FloatNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (IO.FLOAT, {"default": 0.0, "step": 0.01, "defaultInput": False}),
            }
        }

    RETURN_TYPES = (IO.FLOAT,)
    RETURN_NAMES = ("value",)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
    OUTPUT_NODE = True
    DESCRIPTION = "Output a floating-point value from a numeric widget."
    SEARCH_ALIASES = ["float input", "decimal value"]

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
    RETURN_NAMES = ("strings",)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = True
    DESCRIPTION = "Generate a list of stringified numbers from a numeric range."
    SEARCH_ALIASES = ["string range", "sequence text"]

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
    RETURN_NAMES = ("string",)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
    OUTPUT_NODE = True
    DESCRIPTION = "Insert tag text before the existing comma-separated tags."
    SEARCH_ALIASES = ["prefix tags", "prepend prompt tags"]

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
    RETURN_NAMES = ("string",)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
    OUTPUT_NODE = True
    DESCRIPTION = "Append comma-separated tags and remove duplicates."
    SEARCH_ALIASES = ["merge tags", "append prompt tags"]

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
    RETURN_NAMES = ("string",)
    CATEGORY = "Fair/string"

    FUNCTION = "node_function"
    OUTPUT_NODE = True
    DESCRIPTION = "Remove matching tags from a comma-separated tag string."
    SEARCH_ALIASES = ["remove tags", "filter tags"]

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
    RETURN_NAMES = ("string",)
    OUTPUT_NODE = True
    DESCRIPTION = "Show a string in the node UI while forwarding it downstream."
    SEARCH_ALIASES = ["preview text", "display string"]

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
    DESCRIPTION = "Load one text file by path and return its contents and file info."
    SEARCH_ALIASES = ["read text file", "load file text"]

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
    RETURN_NAMES = ("string",)
    DESCRIPTION = "Deduplicate comma-separated tags."
    SEARCH_ALIASES = ["dedupe tags", "unique prompt tags"]

    def node_function(self, string):
        out_string = ""
        tags = string_to_tags(string)
        unique_tags = set(tags)
        out_string = tags_to_string(unique_tags)
        return (out_string,)


class ASCIICharNode:
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
    RETURN_NAMES = ("string",)
    DESCRIPTION = "Convert input text into stylized ASCII art letters."
    SEARCH_ALIASES = ["ascii art", "text banner"]

    # 26个字母的ASCII艺术字
    char_a = """
▒█▀▀█
▒█▄▄█
▒█░▒█
    """

    char_b = """
▒█▀▀▄
▒█▀▀▄
▒█▄▄▀
    """

    char_c = """
▒█▀▀▀
▒█░░░
▒█▄▄▄
    """

    char_d = """
▒█▀▀▄
▒█░▒█
▒█▄▄▀
    """

    char_e = """
▒█▀▀▀
▒█▀▀▀
▒█▄▄▄
    """

    char_f = """
▒█▀▀▀
▒█▀▀▀
▒█░░░
    """

    char_g = """
▒█▀▀▀
▒█░▀█
▒█▄▄█
    """

    char_h = """
▒█░▒█
▒█▀▀█
▒█░▒█
    """

    char_i = """
░▒█░░
░▒█░░
░▒█░░
    """

    char_j = """
░░▒█░
░░▒█░
▒█▄█░
    """

    char_k = """
▒█░▄▀
▒█▀▄░
▒█░▒█
    """

    char_l = """
▒█░░░
▒█░░░
▒█▄▄▄
    """

    char_m = """
▒█▄ ▄█
▒█▒█▒█
▒█░░▒█
    """

    char_n = """
▒█▄▒█
▒██▒█
▒█░▀█
    """

    char_o = """
▒█▀▀█
▒█░▒█
▒█▄▄█
    """

    char_p = """
▒█▀▀█
▒█▄▄█
▒█░░░
    """

    char_q = """
▒█▀▀█
▒█░▒█
▒█▄▀▄
    """

    char_r = """
▒█▀▀█
▒█▄▄▀
▒█░▒█
    """

    char_s = """
▒▄▀▀▀░
░░▀▄░░
▒▄▄▄▀░
    """

    char_t = """
▀▀█▀▀
░▒█░░
░▒█░░
    """

    char_u = """
▒█░▒█
▒█░▒█
▒█▄▄█
    """

    char_v = """
█░░▒█
▒█▒█░
░▒▀░░
    """

    char_w = """
▒█░█░█
▒█░█░█
▒█▄▀▄█
    """

    char_x = """
▀▄░▄▀
░▒█░░
▄▀░▀▄
    """

    char_y = """
█░░▒█
▒█▒█░
░▒█░░
    """

    char_z = """
▒▀▀▀█
░░▄▀░
▒█▄▄▄
    """

    def node_function(self, string):
        # 将输入字符串转换为小写并过滤掉非字母字符
        chars = [c.lower() for c in string if c.isalpha()]

        if not chars:
            return ("",)

        # 获取所有字符的ASCII艺术字
        char_lines = [[], [], []]  # 3行

        for char in chars:
            # 获取对应字母的ASCII艺术字
            char_attr = f"char_{char}"
            if hasattr(self, char_attr):
                char_art = getattr(self, char_attr).strip().split("\n")
                # 将每一行添加到对应的行中
                for i in range(3):
                    if i < len(char_art):
                        char_lines[i].append(char_art[i])

        # 组合所有行，字符之间加空格
        out_string = ""
        for line_parts in char_lines:
            out_string += " ".join(line_parts) + "\n"

        print(out_string)

        return (out_string,)
