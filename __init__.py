from .translate_utility import CLIPTranslatedClass
from .translate_utility import TranslateStringClass
from .load_images_from_folder import LoadImageFromFolderClass
from .text_utility import SaveStringToFolderClass
from .text_utility import FixUTF8StringClass
from .text_utility import StringCombineClass
from .text_utility import StringFieldClass
from .image_utility import DownloadImageClass
from .image_utility import SaveImagesToFolderClass
from .image_utility import SaveImageToFolderClass
from .image_utility import ImageResizeClass

NODE_CLASS_MAPPINGS = {
    "CLIPTranslatedClass": CLIPTranslatedClass,
    "TranslateStringClass": TranslateStringClass,
    "LoadImageFromFolderClass": LoadImageFromFolderClass,
    "SaveStringToFolderClass": SaveStringToFolderClass,
    "FixUTF8StringClass": FixUTF8StringClass,
    "StringCombineClass": StringCombineClass,
    "StringFieldClass": StringFieldClass,
    "DownloadImageClass": DownloadImageClass,
    "SaveImagesToFolderClass": SaveImagesToFolderClass,
    "SaveImageToFolderClass": SaveImageToFolderClass,
    "ImageResizeClass": ImageResizeClass,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTranslatedClass": "CLIP Text Encode Translated",
    "TranslateStringClass": "Translate String",
    "LoadImageFromFolderClass": "Load Image From Folder",
    "SaveStringToFolderClass": "Save String To Folder",
    "FixUTF8StringClass": "Fix UTF-8 String",
    "StringCombineClass": "String Combine",
    "StringFieldClass": "String Field",
    "DownloadImageClass": "Download Image",
    "SaveImagesToFolderClass": "Save Images To Folder",
    "SaveImageToFolderClass": "Save Image To Folder",
    "ImageResizeClass": "Image Resize",
}


WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
