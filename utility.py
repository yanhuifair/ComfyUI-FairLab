class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


class PrintAnyNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": (AlwaysEqualProxy("*"),),
                "log": ("STRING",),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/utility"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    def node_function(self, any, log):
        print(f"log: {log}")
        print(f"type: {type(any)}")
        print(any)

        return ()


class PrintImageNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "log": ("STRING",),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/utility"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    def node_function(self, image, log):
        print(f"log: {log}")
        print(f"shape: {image.shape}")
        print(image)
        return ()


DEFAULT_SCRIPT = "RESULT = (A, B, C, D)"


class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


any = AnyType("*")


class PythonScriptNode:
    def __init__(self):
        pass

    RETURN_TYPES = (any, any, any, any)
    FUNCTION = "run_script"
    OUTPUT_NODE = True
    CATEGORY = "Fair/utility"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "text": ("STRING", {"default": DEFAULT_SCRIPT, "multiline": True}),
                "A": (any, {}),
                "B": (any, {}),
                "C": (any, {}),
                "D": (any, {}),
            },
        }

    def run_script(self, text=DEFAULT_SCRIPT, A=None, B=None, C=None, D=None):
        SCRIPT = text if text is not None and len(text) > 0 else DEFAULT_SCRIPT
        r = compile(SCRIPT, "<string>", "exec")
        ctxt = {"RESULT": None, "A": A, "B": B, "C": C, "D": D}
        eval(r, ctxt)
        return ctxt["RESULT"]
