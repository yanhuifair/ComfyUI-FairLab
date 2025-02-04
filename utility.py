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
        print("----------Print any start----------")
        print(f"log: {log}")
        print(f"type: {type(any)}")
        print(any)
        print("----------Print any end----------")

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
        print("----------Print image start----------")
        print(f"log: {log}")
        print(f"shape: {image.shape}")
        print(image)
        print("----------Print image end----------")

        return ()
