from .market1501 import Market1501
from .dukemtmcreid import DukeMTMCreID
from .cuhk03_detected import CUHK03Detected
from .cuhk03_labeled import CUHK03Labeled
from .dataset_loader import ImageDataset

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'cuhk03-D': CUHK03Detected,
    'cuhk03-L': CUHK03Labeled
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
