from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union, Dict, List, Callable
from numpy import empty

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from data.utils import ImageFolderWithFilenames
from utils import print_once, is_rank_zero
from PIL import Image

__all__ = ['MultiViewImageFolderDataModule']


@dataclass
class MultiViewImageFolderDataModule(LightningDataModule):
    basepath: Union[str, Path]  # Root
    # List of all the cameras to be used for forward processing 
    cameras: List[str]
    dataloader: Dict[str, Any]
    resolution: int = 256  # Image dimension

    def __post_init__(self):
        super().__init__()
        self.path = Path(self.basepath)
        self.stats = {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)}
        self.transform = transforms.Compose([
            t for t in [
                transforms.Resize(self.resolution, InterpolationMode.LANCZOS),
                transforms.CenterCrop(self.resolution),

                # Turning off the horizontal flip as we are working on the two views of the same image  
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.stats['mean'], self.stats['std'], inplace=True),
            ]
        ])
        self.data = {}

    def setup(self, stage: Optional[str] = None):
        for split in ('train', 'validate', 'test'):
            try:
                self.data[split] = MultiViewImageFolderWithFilenames(self.basepath, self.cameras, split,
                                                             transform=self.transform)
            except FileNotFoundError:
                print_once(f'Could not create dataset for split {split}')   

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader('validate')

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader('test')

    def _get_dataloader(self, split: str):
        return DataLoader(self.data[split], **self.dataloader)


@dataclass
class MultiViewImageFolderWithFilenames(Dataset):
    basepath: Union[str, Path]  # Root
    cameras: List[str]
    split: str
    transform: Callable

    def __post_init__(self):
        super().__init__() 

        # Listing all the datasets with the given fixed basepath along with the different camera names for processing. 
        self.image_names = []

        # Defining the folder path for the two views of the camera
        self.camera_view1_path = os.path.join(self.basepath, self.cameras[0], self.split, 'base_class')   # 
        self.camera_view2_path = os.path.join(self.basepath, self.cameras[1], self.split, 'base_class')   #  

        print("Camera view1 path: {}, path exists: {}".format(self.camera_view1_path, os.path.exists(self.camera_view1_path)))
        print("Camera view2 path: {}, path exists: {}".format(self.camera_view2_path, os.path.exists(self.camera_view2_path))) 

        # Reading all the images present in the folder corresponding to each of the camera 
        # So we have to count all the images inside /train/train/ | Here the second train is acting as a class path 
        self.imgs_prefix = os.listdir(self.camera_view1_path)

        # print("imgs prefix: {}".format(self.imgs_prefix))
        self._len = len(self.imgs_prefix)

        print_once(f'Created dataset with {self.cameras}. '
                  f'Lengths are {len(self.cameras)}. Effective dataset length is {self._len}.') 
        # exit()

    def __getitem__(self, index):
        img_name = self.imgs_prefix[index]
        
        # Reading the two images from the two separate folder and we need to process them togather. 
        img_view1_pt = os.path.join(self.camera_view1_path, img_name)
        img_view2_pt = os.path.join(self.camera_view2_path, img_name)

        img_view1 = Image.open(img_view1_pt).convert('RGB')
        img_view2 = Image.open(img_view2_pt).convert('RGB')

        #print_once("img shape: {}".format(img_view1.size))

        img_view1_tformed = self.transform(img_view1)
        img_view2_tformed = self.transform(img_view2)  

    
        # print_once("img transformed 1 shape: {}".format(img_view1_tformed.shape))
        # print_once("img transformed 2 shape: {}".format(img_view2_tformed.shape))

        # x, {'labels': y, 'filenames': self.imgs[i][0]}
        return img_view1_tformed, img_view2_tformed, {'labels': 'Nolabel', 'filenames': img_name}  


    def __len__(self):
        return self._len 
