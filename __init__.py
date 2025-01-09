from .translate_utility import CLIPTranslatedNode
from .translate_utility import TranslateStringNode

from .load_images_from_folder import LoadImageFromFolderNode

from .text_utility import SaveStringToFolderNode
from .text_utility import FixUTF8StringNode
from .text_utility import StringCombineNode
from .text_utility import StringFieldNode
from .text_utility import SequenceStringListNode

from .image_utility import DownloadImageNode
from .image_utility import SaveImagesToFolderNode
from .image_utility import SaveImageToFolderNode
from .image_utility import ImageResizeNode

NODE_CLASS_MAPPINGS = {
    "CLIPTranslatedNode": CLIPTranslatedNode,
    "TranslateStringNode": TranslateStringNode,
    "LoadImageFromFolderNode": LoadImageFromFolderNode,
    "SaveStringToFolderNode": SaveStringToFolderNode,
    "FixUTF8StringNode": FixUTF8StringNode,
    "StringCombineNode": StringCombineNode,
    "StringFieldNode": StringFieldNode,
    "SequenceStringListNode": SequenceStringListNode,
    "DownloadImageNode": DownloadImageNode,
    "SaveImagesToFolderNode": SaveImagesToFolderNode,
    "SaveImageToFolderNode": SaveImageToFolderNode,
    "ImageResizeNode": ImageResizeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTranslatedNode": "CLIP Text Encode Translated",
    "TranslateStringNode": "Translate String",
    "LoadImageFromFolderNode": "Load Image From Folder",
    "SaveStringToFolderNode": "Save String To Folder",
    "FixUTF8StringNode": "Fix UTF-8 String",
    "StringCombineNode": "String Combine",
    "StringFieldNode": "String Field",
    "SequenceStringListNode": "Sequence String List",
    "DownloadImageNode": "Download Image",
    "SaveImagesToFolderNode": "Save Images To Folder",
    "SaveImageToFolderNode": "Save Image To Folder",
    "ImageResizeNode": "Image Resize",
}


WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
