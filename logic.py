import math
from comfy.comfy_types.node_typing import IO


class MaxNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": (IO.ANY,),
                "b": (IO.ANY,),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/logic"
    RETURN_TYPES = (IO.ANY,)

    def node_function(self, a, b):
        out_value = max(a, b)
        return (out_value,)


class MinNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": (IO.ANY,),
                "b": (IO.ANY,),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/logic"
    RETURN_TYPES = (IO.ANY,)

    def node_function(self, a, b):
        out_value = min(a, b)
        return (out_value,)


class AddNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": (IO.FLOAT,),
                "b": (IO.FLOAT,),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/logic"
    RETURN_TYPES = (IO.FLOAT,)

    def node_function(self, a, b):
        out_value = a + b
        return (out_value,)


class SubtractNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": (IO.FLOAT,),
                "b": (IO.FLOAT,),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/logic"
    RETURN_TYPES = (IO.FLOAT,)

    def node_function(self, a, b):
        out_value = a - b
        return (out_value,)


class MultiplyNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": (IO.FLOAT,),
                "b": (IO.FLOAT,),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/logic"
    RETURN_TYPES = (IO.FLOAT,)

    def node_function(self, a, b):
        out_value = a * b
        return (out_value,)


class DivideNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": (IO.FLOAT,),
                "b": (IO.FLOAT,),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/logic"
    RETURN_TYPES = (IO.FLOAT,)

    def node_function(self, a, b):
        out_value = a / b
        return (out_value,)


class NumberNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number": (IO.FLOAT,),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/logic"
    RETURN_TYPES = (IO.INT, IO.FLOAT, IO.FLOAT, IO.FLOAT, IO.FLOAT, IO.FLOAT, IO.FLOAT, IO.FLOAT, IO.FLOAT)
    RETURN_NAMES = ("float", "int", "absolute", "round", "ceil", "floor", "sqrt", "exp", "log")

    def node_function(self, number):
        out_int = int(number)
        out_float = float(out_int)
        out_abs = abs(out_float)
        out_round = round(out_float)
        out_ceil = math.ceil(out_float)
        out_floor = math.floor(out_float)
        out_sqrt = math.sqrt(out_abs)
        out_exp = math.exp(out_float)
        out_log = math.log(out_float)
        return (out_int, out_float, out_abs, out_round, out_ceil, out_floor, out_sqrt, out_exp, out_log)


class FloatToIntNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (IO.FLOAT,),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/logic"
    RETURN_TYPES = (IO.INT,)

    def node_function(self, value):
        out_value = int(value)
        return (out_value,)


class IntToFloatNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (IO.INT,),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/logic"
    RETURN_TYPES = (IO.FLOAT,)

    def node_function(self, value):
        out_value = int(value)
        return (out_value,)


class IfNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "true_value": (IO.ANY,),
                "false_value": (IO.ANY,),
                "condition": (IO.BOOLEAN, {"defaultInput": True}),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/logic"
    RETURN_TYPES = (IO.ANY,)

    def node_function(self, true_value, false_value, condition):
        if condition:
            return (true_value,)
        else:
            return (false_value,)
