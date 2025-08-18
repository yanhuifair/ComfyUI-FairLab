from .utility import any

DEFAULT_SCRIPT = "RESULT = (A, B, C, D)"


class PythonScriptNode:

    def __init__(self):
        pass

    RETURN_TYPES = (any, any, any, any)
    FUNCTION = "run_script"
    OUTPUT_NODE = True
    CATEGORY = "Fair/utility"

    @classmethod
    def INPUT_TYPES(cls):
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
        code = compile(SCRIPT, "<string>", "exec")
        dic = {"RESULT": None, "A": A, "B": B, "C": C, "D": D}
        eval(code, dic)
        return dic["RESULT"]
