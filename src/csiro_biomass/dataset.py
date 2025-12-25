"""Dataset class for CSIRO biomass dataset."""

import os

import albumentations as A
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from csiro_biomass.utils.io import read_csv


class CSIRO(Dataset):
    """CSIRO biomass dataset."""

    def __init__(self, config, feature_extractor=None):

        d_data = read_csv(config.dataset.train)
        d_data = self.data_cleanup(d_data)
        d_data = self.target_transform(d_data)

        self.data_fold = {}
        skf = StratifiedKFold(
            n_splits=config.dataset.kfolds, shuffle=True, random_state=config.general.seed
        )
        for i_fold, (t_index, v_index) in enumerate(skf.split(d_data, d_data.outlier)):
            train = d_data.iloc[t_index]
            valid = d_data.iloc[v_index]
            self.data_fold[i_fold] = {"train": (train, len(train)), "valid": (valid, len(valid))}

        self.split_name = None
        self.data = None
        self.length = None
        self.img_dir_path = config.general.img_dir
        self.feature_extractor = feature_extractor
        self.transform = self.get_tta()

    @staticmethod
    def data_cleanup(d_data):
        """Clean up dataset."""
        d_data = d_data.pivot(
            index="image_path", columns="target_name", values="target"
        ).reset_index()
        d_data.columns.name = None
        d_train_selected = d_data[
            ["image_path", "Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]
        ].copy()

        clover_q1, dead_q1, green_q1 = (
            d_train_selected[["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]].quantile(0.25).tolist()
        )
        clover_q3, dead_q3, green_q3 = (
            d_train_selected[["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]].quantile(0.75).tolist()
        )
        max_clover = clover_q3 + ((clover_q3 - clover_q1) * 1.5)
        max_dead = dead_q3 + ((dead_q3 - dead_q1) * 1.5)
        max_green = green_q3 + ((green_q3 - green_q1) * 1.5)
        d_train_selected["is_clover_outlier"] = d_train_selected.Dry_Clover_g.apply(
            lambda x: 1 if x > max_clover else 0
        )
        d_train_selected["is_dead_outlier"] = d_train_selected.Dry_Dead_g.apply(
            lambda x: 1 if x > max_dead else 0
        )
        d_train_selected["is_green_outlier"] = d_train_selected.Dry_Green_g.apply(
            lambda x: 1 if x > max_green else 0
        )

        d_train_selected["outlier"] = d_train_selected.apply(
            lambda row: row.is_clover_outlier or row.is_dead_outlier or row.is_green_outlier, axis=1
        )

        return d_train_selected

    def target_transform(self, d_data, method="log1p"):
        """Transform target variable."""

        if method == "log1p":
            d_data.iloc[:, 1:4] = d_data.iloc[:, 1:4].apply(np.log1p)

        return d_data

    def set_fold(self, i_fold):
        """Select dataset fold."""
        self.data = self.data_fold[i_fold]

    def set_split(self, split="train"):
        """Select dataset partition.

        Args:
            split (str, optional): split partition. Defaults to "train".
        """
        self.split_name = split
        self.data, self.length = self.data[split]

    def get_tta(self):
        """Test Time augmentation"""
        transforms = [
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1.0),
        ]
        return transforms

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        img_full_path = os.path.join(self.img_dir_path, img_path)
        img = Image.open(img_full_path)
        img_arr = np.array(img)

        if (self.split_name == "valid") and self.transform:
            emb_list = []
            for aug in self.transform:
                img_arr = aug(image=img_arr)["image"]
                emb = self.feature_extractor(img_arr)
                emb = emb.unsqueeze(0)
                emb_list.append(emb)

            x = torch.cat(emb_list, dim=0)

            y = np.array(self.data.iloc[idx, 1:4].tolist())
        else:
            x = self.feature_extractor(img_arr)
            y = np.array(self.data.iloc[idx, 1:4].tolist())

        return x, y

    def __len__(self):
        return self.length
