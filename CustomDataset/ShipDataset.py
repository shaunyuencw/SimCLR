import os.path
import pandas as pd
import numpy as np
from typing import Any, Callable, Iterable, Optional, Tuple, TypeVar, cast
from torch.utils.data import Dataset
from PIL import Image

class ShipDataset(Dataset):

    base_folder = "ship_data"
    splits = ("train", "train+unlabeled", "unlabeled", "test")

    def __init__(
        self,
        root: str,
        split: str = "train",
        l_img_pc: int = 0,
        ul_img_pc: int = 0,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            root: str -> Root folder
            split: str -> Type of split
            transform: [Optional] transforms
            l_img_pc: int ->  number of label images per class
            ul_img_pc: int -> number of unlabeled iamges per class

        Return:
            None
        """
        self.root = root
        self.split = split
        assert split in self.splits, "Invalid split type"


        self.transform = transform
        if self.split == "train+unlabeled":
            self.data, self.labels = self.__loadfile(l_img_pc, True)
            unlabeled_data, _ = self.__loadfile(ul_img_pc, False)
            self.data = pd.concat((self.data, unlabeled_data))
            self.labels = np.concatenate((self.labels, np.asarray([-1] * unlabeled_data.shape[0])))

        elif self.split == "unlabeled":
            self.data, _ = self.__loadfile(ul_img_pc, False)
            self.labels = np.asarray([-1] * self.data.shape[0])

        else: # "train", "val" and "test"
            self.data, self.labels = self.__loadfile(l_img_pc, True)

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img_path = self.data.iloc[index][0]
        #print(f"+ __getitem__() -> {img_path}")
        image = Image.open(img_path)
        y_label = int(self.data.iloc[index, 1])

        if self.transform:
            #print(f"+ Transforming image")
            aug_images = self.transform(image)
        
        return (aug_images, y_label)

    # TODO Set start index to prevent overlap
    def __loadfile(self, img_pc:int=0, require_label:bool=True):
        """
        Args:
            img_pc: int -> Images per class
            require_label: bool -> Whether to include labels
        Returns:
            image_df: Dataframe of image_paths and labels
            labels: np_array of labels
        """

        image_df = pd.DataFrame(columns=["image", 'label'])
        training = ("train", "train+unlabeled", "unlabeled")

        if self.split in training:
            data_folder = os.path.join(self.root, self.base_folder, "train")
        else:
            data_folder = os.path.join(self.root, self.base_folder, "val")

        labels = []

        
        #! Change self.split to train, unlabeled_train, validation, test?
        self.classes = [f.path for f in os.scandir(data_folder) if f.is_dir()]

        for index, class_path in  enumerate(self.classes):
            images = [f.path for f in os.scandir(class_path) if f.is_file()]

            # If 
            if img_pc <= 0:
                img_pc = len(images)

            for idx, image_path in enumerate(images):
                if idx >= img_pc:
                    break

                if require_label:
                    image_df.loc[len(image_df.index)] = [image_path, index]
                    labels.append(index)
                else:
                    image_df.loc[len(image_df.index)] = [image_path, -1]
                    labels.append(-1)

        return image_df, np.array(labels)