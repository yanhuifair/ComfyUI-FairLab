from comfy.comfy_types.node_typing import IO

DEFAULT_SCRIPT = "RESULT = (A, B, C, D)"


class PythonScriptNode:

    def __init__(self):
        pass

    RETURN_TYPES = (IO.ANY, IO.ANY, IO.ANY, IO.ANY)
    FUNCTION = "run_script"
    OUTPUT_NODE = True
    CATEGORY = "Fair/utility"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "text": (IO.STRING, {"default": DEFAULT_SCRIPT, "multiline": True}),
                "A": (IO.ANY, {}),
                "B": (IO.ANY, {}),
                "C": (IO.ANY, {}),
                "D": (IO.ANY, {}),
            },
        }

    def run_script(self, text=DEFAULT_SCRIPT, A=None, B=None, C=None, D=None):
        SCRIPT = text if text is not None and len(text) > 0 else DEFAULT_SCRIPT
        code = compile(SCRIPT, "<string>", "exec")
        dic = {"RESULT": None, "A": A, "B": B, "C": C, "D": D}
        eval(code, dic)
        return dic["RESULT"]
