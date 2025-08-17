from .utility import AlwaysEqualProxy


class MaxNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number0": ("FLOAT",),
                "number1": ("FLOAT",),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/logic"
    RETURN_TYPES = ("FLOAT",)

    def node_function(self, number0, number1):
        out_value = max(number0, number1)
        return (out_value,)


class MinNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number0": ("FLOAT",),
                "number1": ("FLOAT",),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/logic"
    RETURN_TYPES = ("FLOAT",)

    def node_function(self, number0, number1):
        out_value = min(number0, number1)
        return (out_value,)


class AddNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number0": ("FLOAT",),
                "number1": ("FLOAT",),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/logic"
    RETURN_TYPES = ("FLOAT",)

    def node_function(self, number0, number1):
        out_value = number0 + number1
        return (out_value,)


class SubtractNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number0": ("FLOAT",),
                "number1": ("FLOAT",),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/logic"
    RETURN_TYPES = ("FLOAT",)

    def node_function(self, number0, number1):
        out_value = number0 - number1
        return (out_value,)


class MultiplyNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number0": ("FLOAT",),
                "number1": ("FLOAT",),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/logic"
    RETURN_TYPES = ("FLOAT",)

    def node_function(self, number0, number1):
        out_value = number0 * number1
        return (out_value,)


class DivideNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number0": ("FLOAT",),
                "number1": ("FLOAT",),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/logic"
    RETURN_TYPES = ("FLOAT",)

    def node_function(self, number0, number1):
        out_value = number0 / number1
        return (out_value,)


class IfNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "true_value": (AlwaysEqualProxy("*"),),
                "false_value": (AlwaysEqualProxy("*"),),
                "condition": ("BOOL",),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/logic"
    RETURN_TYPES = (AlwaysEqualProxy("*"),)

    def node_function(self, true_value, false_value, condition):
        if condition:
            return (true_value,)
        else:
            return (false_value,)
