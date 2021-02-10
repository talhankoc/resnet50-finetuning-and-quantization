from typing import List, Tuple
import os
import json

import numpy as np
import torch
from PIL import Image

DATA_DIR = 'data/Images/'
NAMES_TO_IDS_FN = "data/names_to_ids.json"


class IndoorDataset(torch.utils.data.Dataset):

    def __init__(self, file_names: List[str]):
        self.file_names = file_names

        # e.g. 'airport_inside/airport...0001.jpg' -> 'airport_inside'
        labels = [fn[:fn.find('/')] for fn in self.file_names]
        with open(NAMES_TO_IDS_FN) as f:
            label_to_ids = json.load(f)
        self.label_ids = [label_to_ids[label] for label in labels]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        fn = self.file_names[index]
        raw_image_fn = f"{DATA_DIR}{fn}"
        # same as `fn` but with `.jpg` replaced with `.npy`
        feature_fn = f"{DATA_DIR}{fn[:fn.find('.')]}.npy"

        if not os.path.exists(feature_fn):
            # We have to transform the
            image = Image.open(raw_image_fn)
            image = self._apply_image_transformations(image)
            feature = np.asarray(image)
            feature = feature.T  # channel needs to be the first dimension
            np.save(feature_fn, feature)  # save so that this isn't calculated every time we run
        else:
            feature = np.load(feature_fn)
        feature = torch.Tensor(feature)

        label_id = self.label_ids[index]

        return feature, label_id

    @staticmethod
    def _apply_image_transformations(image):
        return image.resize((224, 224)).convert('RGB')
