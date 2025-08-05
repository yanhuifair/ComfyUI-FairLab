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


class MaxNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int1": ("INT",),
                "int2": ("INT",),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/utility"
    RETURN_TYPES = ("INT",)

    def node_function(self, int1, int2):
        max_value = max(int1, int2)
        return (max_value,)


class MinNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int1": ("INT",),
                "int2": ("INT",),
            },
        }

    FUNCTION = "node_function"
    CATEGORY = "Fair/utility"
    RETURN_TYPES = ("INT",)

    def node_function(self, int1, int2):
        min_value = min(int1, int2)
        return (min_value,)
