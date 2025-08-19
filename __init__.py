from .translate_utility import StringTranslateNode

from .text_utility import SaveStringToDirectoryNode
from .text_utility import LoadStringFromDirectoryNode
from .text_utility import FixUTF8StringNode
from .text_utility import StringCombineNode
from .text_utility import StringNode
from .text_utility import IntNode
from .text_utility import FloatNode
from .text_utility import SequenceStringListNode
from .text_utility import PrependTagsNode
from .text_utility import AppendTagsNode
from .text_utility import BlacklistTagsNode
from .text_utility import ShowStringNode
from .text_utility import LoadStringNode
from .text_utility import RemoveDuplicateTagsNode

from .image_utility import DownloadImageNode
from .image_utility import SaveImageToDirectoryNode
from .image_utility import ImageResizeNode
from .image_utility import VideoToImageNode
from .image_utility import ImageToVideoNode
from .image_utility import LoadImageFromURLNode
from .image_utility import LoadImageFromDirectoryNode
from .image_utility import FillAlphaNode
from .image_utility import ImageToBase64Node
from .image_utility import Base64ToImageNode
from .image_utility import OutpaintingPadNode
from .image_utility import ImageSizeNode

from .utility import PrintAnyNode
from .utility import PrintImageNode
from .utility import LoraLoaderDualNode

from .script import PythonScriptNode

from .logic import MaxNode
from .logic import MinNode
from .logic import AddNode
from .logic import SubtractNode
from .logic import MultiplyNode
from .logic import DivideNode
from .logic import NumberNode
from .logic import IfNode
from .logic import FloatToIntNode
from .logic import IntToFloatNode


from .ollama import OllamaNode
from .ollama import OllamaClientNode


NODE_CLASS_MAPPINGS = {
    "StringTranslateNode": StringTranslateNode,
    "LoadImageFromDirectoryNode": LoadImageFromDirectoryNode,
    "FillAlphaNode": FillAlphaNode,
    "SaveStringToDirectoryNode": SaveStringToDirectoryNode,
    "LoadStringFromDirectoryNode": LoadStringFromDirectoryNode,
    "FixUTF8StringNode": FixUTF8StringNode,
    "StringCombineNode": StringCombineNode,
    "StringNode": StringNode,
    "IntNode": IntNode,
    "FloatNode": FloatNode,
    "SequenceStringListNode": SequenceStringListNode,
    "PrependTagsNode": PrependTagsNode,
    "AppendTagsNode": AppendTagsNode,
    "BlacklistTagsNode": BlacklistTagsNode,
    "LoadStringNode": LoadStringNode,
    "RemoveDuplicateTagsNode": RemoveDuplicateTagsNode,
    "ShowStringNode": ShowStringNode,
    "DownloadImageNode": DownloadImageNode,
    "SaveImageToDirectoryNode": SaveImageToDirectoryNode,
    "ImageResizeNode": ImageResizeNode,
    "VideoToImageNode": VideoToImageNode,
    "ImageToVideoNode": ImageToVideoNode,
    "LoadImageFromURLNode": LoadImageFromURLNode,
    "PrintAnyNode": PrintAnyNode,
    "PrintImageNode": PrintImageNode,
    "LoraLoaderDualNode": LoraLoaderDualNode,
    "PythonScriptNode": PythonScriptNode,
    "MaxNode": MaxNode,
    "MinNode": MinNode,
    "AddNode": AddNode,
    "SubtractNode": SubtractNode,
    "MultiplyNode": MultiplyNode,
    "DivideNode": DivideNode,
    "NumberNode": NumberNode,
    "IfNode": IfNode,
    "FloatToIntNode": FloatToIntNode,
    "IntToFloatNode": IntToFloatNode,
    "ImageToBase64Node": ImageToBase64Node,
    "OutpaintingPadNode": OutpaintingPadNode,
    "ImageSizeNode": ImageSizeNode,
    "Base64ToImageNode": Base64ToImageNode,
    "OllamaNode": OllamaNode,
    "OllamaClientNode": OllamaClientNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringTranslateNode": "String Translate",
    "LoadImageFromDirectoryNode": "Load Image From Directory",
    "FillAlphaNode": "Fill Alpha",
    "SaveStringToDirectoryNode": "Save String To Directory",
    "LoadStringFromDirectoryNode": "Load String From Directory",
    "FixUTF8StringNode": "Fix UTF-8 String",
    "StringCombineNode": "String Combine",
    "StringNode": "String",
    "IntNode": "Int",
    "FloatNode": "Float",
    "SequenceStringListNode": "Sequence String List",
    "PrependTagsNode": "Prepend Tags",
    "AppendTagsNode": "Append Tags",
    "BlacklistTagsNode": "Blacklist Tags",
    "LoadStringNode": "Load String",
    "RemoveDuplicateTagsNode": "Remove Duplicate Tags",
    "ShowStringNode": "Show String",
    "DownloadImageNode": "Download Image",
    "SaveImageToDirectoryNode": "Save Image To Directory",
    "ImageResizeNode": "Image Resize",
    "VideoToImageNode": "Video To Image",
    "ImageToVideoNode": "Image To Video",
    "LoadImageFromURLNode": "Load Image From URL",
    "PrintAnyNode": "Print Any",
    "PrintImageNode": "Print Image",
    "LoraLoaderDualNode": "Load LoRA Dual",
    "PythonScriptNode": "Python Script",
    "MaxNode": "Max",
    "MinNode": "Min",
    "AddNode": "Add",
    "SubtractNode": "Subtract",
    "MultiplyNode": "Multiply",
    "DivideNode": "Divide",
    "NumberNode": "Number",
    "IfNode": "If",
    "FloatToIntNode": "Float To Int",
    "IntToFloatNode": "Int To Float",
    "ImageToBase64Node": "Image To Base64",
    "Base64ToImageNode": "Base64 To Image",
    "OutpaintingPadNode": "Outpainting Pad",
    "ImageSizeNode": "Image Size",
    "OllamaNode": "🦙 Ollama",
    "OllamaClientNode": "🦙 Ollama Client",
}


WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
