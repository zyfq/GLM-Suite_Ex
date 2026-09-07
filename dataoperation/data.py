"""Load, split, normalize, and iterate over CSV datasets."""

import json
from pathlib import Path

import numpy as np


class DataSet:
    """Read the first CSV in ``data``, then iterate over training samples.

    Each item returned by the iterator is ``(features, label)``.  The last CSV
    column is treated as the label, and the other columns are treated as
    numeric features.
    """
    def __init__(self, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2,
                 seed=42, data_dir=None):
        ratios = (train_ratio, validation_ratio, test_ratio)
        if any(ratio < 0 for ratio in ratios) or not np.isclose(sum(ratios), 1):
            raise ValueError("分割比例必须非负，并且总和为 1")

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
        self.train_data, self.validation_data, self.test_data = self.spilt_data()
        self._index = 0
        self._current_data = self.train_data

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self._current_data[0]):
            raise StopIteration
        sample = (
            self._current_data[0][self._index],
            self._current_data[1][self._index],
        )
        self._index += 1
        return sample

    def _read_csv(self):
        csv_files = sorted(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"目录中没有找到 CSV 文件: {self.data_dir}")

        try:
            raw_data = np.genfromtxt(csv_files[0], delimiter=",", skip_header=1)
        except (OSError, ValueError) as error:
            raise ValueError(f"无法读取 CSV 文件: {csv_files[0]}") from error

        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)
        if raw_data.shape[0] == 0 or raw_data.shape[1] < 2:
            raise ValueError("CSV 至少需要一行数据、一个特征列和一个标签列")
        if not np.isfinite(raw_data).all():
            raise ValueError("CSV 中包含缺失值或非数字数据")
        return raw_data

    def _split_indices(self, sample_count):
        indices = np.arange(sample_count)
        np.random.default_rng(self.seed).shuffle(indices)
        train_end = int(sample_count * self.train_ratio)
        validation_end = train_end + int(sample_count * self.validation_ratio)
        return indices[:train_end], indices[train_end:validation_end], indices[validation_end:]

    @staticmethod
    def _normalize(features, train_features):
        minimum = train_features.min(axis=0)
        maximum = train_features.max(axis=0)
        scale = np.where(maximum == minimum, 1.0, maximum - minimum)
        return (features - minimum) / scale

    def spilt_data(self):
        """Return normalized ``(features, labels)`` for train/validation/test."""
        raw_data = self._read_csv()
        features = raw_data[:, :-1].astype(float)
        labels = raw_data[:, -1]
        train_indices, validation_indices, test_indices = self._split_indices(len(raw_data))

        train_features = features[train_indices]
        self.feature_minimum = train_features.min(axis=0)
        self.feature_maximum = train_features.max(axis=0)
        normalized_features = self._normalize(features, train_features)
        return (
            (normalized_features[train_indices], labels[train_indices]),
            (normalized_features[validation_indices], labels[validation_indices]),
            (normalized_features[test_indices], labels[test_indices]),
        )

    def save_preprocessing(self, path):
        #TODO 保存归一化参数到代码目录之外的 JSON 文件，供新数据预测时复用
        path = Path(path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            path = project_root / "runs" / "weights" / path
        path = path.with_suffix(".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        content = {
            "feature_minimum": {
                f"feature_{index + 1}": float(value)
                for index, value in enumerate(self.feature_minimum)
            },
            "feature_maximum": {
                f"feature_{index + 1}": float(value)
                for index, value in enumerate(self.feature_maximum)
            },
        }
        path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load_preprocessing(self, path):
        #TODO 从 JSON 文件读取归一化参数，用于还原训练阶段的预处理流程
        content = json.loads(Path(path).read_text(encoding="utf-8"))
        if "feature_minimum" not in content or "feature_maximum" not in content:
            raise ValueError("预处理文件缺少 feature_minimum 或 feature_maximum")
        minimum = np.asarray(list(content["feature_minimum"].values()), dtype=float)
        maximum = np.asarray(list(content["feature_maximum"].values()), dtype=float)
        if minimum.shape != self.feature_minimum.shape or maximum.shape != self.feature_maximum.shape:
            raise ValueError("归一化参数的维度与当前数据集不一致")
        self.feature_minimum = minimum
        self.feature_maximum = maximum
        return self

    def transform_new_data(self, raw_features):
        #TODO 使用已保存的归一化参数处理新数据，使其与训练数据分布一致
        raw_features = np.asarray(raw_features, dtype=float)
        if raw_features.ndim == 1:
            raw_features = raw_features.reshape(1, -1)
        if raw_features.shape[1] != self.feature_minimum.size:
            raise ValueError("新数据的特征列数与训练数据不一致")
        if not np.isfinite(raw_features).all():
            raise ValueError("新数据中包含缺失值或非数字数据")
        scale = np.where(self.feature_maximum == self.feature_minimum, 1.0,
                         self.feature_maximum - self.feature_minimum)
        return (raw_features - self.feature_minimum) / scale

    def use_split(self, split_name="train"):
        """Choose which split subsequent iteration should traverse."""
        splits = {
            "train": self.train_data,
            "validation": self.validation_data,
            "test": self.test_data,
        }
        try:
            self._current_data = splits[split_name]
        except KeyError as error:
            raise ValueError("split_name 必须是 train、validation 或 test") from error
        self._index = 0
        return self


if __name__ == '__main__':
    file = DataSet(train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2,
                 seed=42, data_dir="../data")
