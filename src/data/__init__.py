import sys

from utils import to_dataclass_cfg
from .nodata import *
from .imagefolder import *
from .multiimagefolder import *
from .multiviewimagefolder import *

def get_datamodule(name: str, **kwargs) -> LightningDataModule:
    print("Modules for this particular class: {}, name: {}, __name__:{}".format(sys.modules[__name__], name, __name__))
    # exit() 
    cls = getattr(sys.modules[__name__], name)
    print("CLS value: {}".format(cls))
    print("kwargs where camera is missing: {}".format(kwargs))
    return cls(**to_dataclass_cfg(kwargs, cls))
