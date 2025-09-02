from .nodes.translate_utility import StringTranslateNode

from .nodes.text_utility import SaveStringToDirectoryNode
from .nodes.text_utility import LoadStringFromDirectoryNode
from .nodes.text_utility import FixUTF8StringNode
from .nodes.text_utility import StringAppendNode
from .nodes.text_utility import StringNode
from .nodes.text_utility import IntNode
from .nodes.text_utility import FloatNode
from .nodes.text_utility import RangeStringNode
from .nodes.text_utility import PrependTagsNode
from .nodes.text_utility import AppendTagsNode
from .nodes.text_utility import ExcludeTagsNode
from .nodes.text_utility import ShowStringNode
from .nodes.text_utility import LoadStringNode
from .nodes.text_utility import UniqueTagsNode

from .nodes.image_utility import DownloadImageNode
from .nodes.image_utility import SaveImageToDirectoryNode
from .nodes.image_utility import ImageResizeNode
from .nodes.image_utility import VideoToImageNode
from .nodes.image_utility import ImageToVideoNode
from .nodes.image_utility import LoadImageFromURLNode
from .nodes.image_utility import LoadImageFromDirectoryNode
from .nodes.image_utility import FillAlphaNode
from .nodes.image_utility import ImageToBase64Node
from .nodes.image_utility import Base64ToImageNode
from .nodes.image_utility import OutpaintingPadNode
from .nodes.image_utility import ImageSizeNode
from .nodes.image_utility import ImagesRangeNode
from .nodes.image_utility import ImagesIndexNode
from .nodes.image_utility import ImagesCatNode

from .nodes.utility import PrintAnyNode
from .nodes.utility import PrintImageNode
from .nodes.utility import LoraLoaderDualNode

from .nodes.script import PythonScriptNode

from .nodes.logic import MaxNode
from .nodes.logic import MinNode
from .nodes.logic import AddNode
from .nodes.logic import SubtractNode
from .nodes.logic import MultiplyNode
from .nodes.logic import DivideNode
from .nodes.logic import NumberNode
from .nodes.logic import IfNode
from .nodes.logic import FloatToIntNode
from .nodes.logic import IntToFloatNode


NODE_CLASS_MAPPINGS = {
    "StringTranslateNode": StringTranslateNode,
    "LoadImageFromDirectoryNode": LoadImageFromDirectoryNode,
    "FillAlphaNode": FillAlphaNode,
    "SaveStringToDirectoryNode": SaveStringToDirectoryNode,
    "LoadStringFromDirectoryNode": LoadStringFromDirectoryNode,
    "FixUTF8StringNode": FixUTF8StringNode,
    "StringAppendNode": StringAppendNode,
    "StringNode": StringNode,
    "IntNode": IntNode,
    "FloatNode": FloatNode,
    "RangeStringNode": RangeStringNode,
    "PrependTagsNode": PrependTagsNode,
    "AppendTagsNode": AppendTagsNode,
    "ExcludeTagsNode": ExcludeTagsNode,
    "LoadStringNode": LoadStringNode,
    "UniqueTagsNode": UniqueTagsNode,
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
    "ImagesRangeNode": ImagesRangeNode,
    "ImagesIndexNode": ImagesIndexNode,
    "ImagesCatNode": ImagesCatNode,
    "Base64ToImageNode": Base64ToImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringTranslateNode": "String Translate",
    "LoadImageFromDirectoryNode": "Load Image From Directory",
    "FillAlphaNode": "Fill Alpha",
    "SaveStringToDirectoryNode": "Save String To Directory",
    "LoadStringFromDirectoryNode": "Load String From Directory",
    "FixUTF8StringNode": "Fix UTF-8 String",
    "StringAppendNode": "String Append",
    "StringNode": "String",
    "IntNode": "Int",
    "FloatNode": "Float",
    "RangeStringListNode": "Range String",
    "PrependTagsNode": "Prepend Tags",
    "AppendTagsNode": "Append Tags",
    "ExcludeTagsNode": "Exclude Tags",
    "LoadStringNode": "Load String",
    "UniqueTagsNode": "Unique Tags",
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
    "ImagesRangeNode": "Images Range",
    "ImagesIndexNode": "Images Index",
    "ImagesCatNode": "Images Cat",
}


WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
