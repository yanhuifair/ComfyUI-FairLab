from .translate_utility import CLIPTranslatedNode
from .translate_utility import TranslateStringNode

from .text_utility import SaveStringToDirectoryNode
from .text_utility import FixUTF8StringNode
from .text_utility import StringCombineNode
from .text_utility import StringFieldNode
from .text_utility import SequenceStringListNode

from .image_utility import DownloadImageNode
from .image_utility import SaveImageToDirectoryNode
from .image_utility import ImageResizeNode
from .image_utility import VideoToImagesNode
from .image_utility import ImagesToVideoNode
from .image_utility import LoadImageFromURLNode
from .image_utility import LoadImageFromDirectoryNode

from .utility import PrintAnyNode
from .utility import PrintImageNode

NODE_CLASS_MAPPINGS = {
    "CLIPTranslatedNode": CLIPTranslatedNode,
    "TranslateStringNode": TranslateStringNode,
    "LoadImageFromDirectoryNode": LoadImageFromDirectoryNode,
    "SaveStringToDirectoryNode": SaveStringToDirectoryNode,
    "FixUTF8StringNode": FixUTF8StringNode,
    "StringCombineNode": StringCombineNode,
    "StringFieldNode": StringFieldNode,
    "SequenceStringListNode": SequenceStringListNode,
    "DownloadImageNode": DownloadImageNode,
    "SaveImageToDirectoryNode": SaveImageToDirectoryNode,
    "ImageResizeNode": ImageResizeNode,
    "VideoToImagesNode": VideoToImagesNode,
    "ImagesToVideoNode": ImagesToVideoNode,
    "LoadImageFromURLNode": LoadImageFromURLNode,
    "PrintAnyNode": PrintAnyNode,
    "PrintImageNode": PrintImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTranslatedNode": "CLIP Text Encode Translated",
    "TranslateStringNode": "Translate String",
    "LoadImageFromDirectoryNode": "Load Image From Directory",
    "SaveStringToDirectoryNode": "Save String To Directory",
    "FixUTF8StringNode": "Fix UTF-8 String",
    "StringCombineNode": "String Combine",
    "StringFieldNode": "String Field",
    "SequenceStringListNode": "Sequence String List",
    "DownloadImageNode": "Download Image",
    "SaveImageToDirectoryNode": "Save Image To Directory",
    "ImageResizeNode": "Image Resize",
    "VideoToImagesNode": "Video To Images",
    "ImagesToVideoNode": "Images To Video",
    "LoadImageFromURLNode": "Load Image From URL",
    "PrintAnyNode": "Print Any",
    "PrintImageNode": "Print Image",
}


WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
