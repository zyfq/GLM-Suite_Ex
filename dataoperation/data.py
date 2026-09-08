"""Load, split, normalize, and iterate over CSV datasets."""

import json
import os
from pathlib import Path

import numpy as np


#数据集统一存放在项目源码之外的本机目录，避免代码仓库存储大文件，可被多个项目共享
DEFAULT_DATASET_ROOT = Path(r"D:\zyf\data")
DATASET_ROOT_ENV = "DATASET_ROOT"


def _resolve_dataset_root(data_dir=None):
    #TODO 解析数据集根目录：显式参数 > 环境变量 DATASET_ROOT > 默认本机数据集目录
    if data_dir is not None:
        return Path(data_dir)
    env_root = os.environ.get(DATASET_ROOT_ENV)
    if env_root:
        return Path(env_root)
    return DEFAULT_DATASET_ROOT


class DataSet:
    """Read a CSV from the shared dataset directory and iterate over training samples.

    Each item returned by the iterator is ``(features, label)``.  The last CSV
    column is treated as the label, and the other columns are treated as
    numeric features.
    """
    def __init__(self, csv_name=None, train_ratio=0.6, validation_ratio=0.2,
                 test_ratio=0.2, seed=42, data_dir=None):
        ratios = (train_ratio, validation_ratio, test_ratio)
        if any(ratio < 0 for ratio in ratios) or not np.isclose(sum(ratios), 1):
            raise ValueError("分割比例必须非负，并且总和为 1")

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.data_dir = _resolve_dataset_root(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"数据集目录不存在: {self.data_dir}，"
                f"可设置环境变量 {DATASET_ROOT_ENV} 指向实际目录"
            )
        self.csv_path = self._resolve_csv_path(csv_name)
        self.train_data, self.validation_data, self.test_data = self.spilt_data()
        self._index = 0
        self._current_data = self.train_data

    def _resolve_csv_path(self, csv_name):
        #TODO 在共享数据集目录中定位指定的 CSV 文件，支持子目录
        if csv_name is None:
            csv_files = sorted(self.data_dir.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"数据集目录中没有 CSV 文件: {self.data_dir}")
            return csv_files[0]

        if Path(csv_name).is_absolute():
            raise ValueError("csv_name 必须是相对于数据集目录的相对路径")
        candidate = self.data_dir / csv_name
        if candidate.exists():
            return candidate

        matches = sorted(path for path in self.data_dir.rglob("*.csv")
                         if path.name == Path(csv_name).name)
        if not matches:
            available = "\n".join(str(path.relative_to(self.data_dir))
                                  for path in sorted(self.data_dir.rglob("*.csv")))
            raise FileNotFoundError(
                f"数据集目录中没有找到: {csv_name}\n当前可用的数据集:\n{available}"
            )
        return matches[0]

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
        if not self.csv_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.csv_path}")

        try:
            raw_data = np.genfromtxt(self.csv_path, delimiter=",", skip_header=1)
        except (OSError, ValueError) as error:
            raise ValueError(f"无法读取 CSV 文件: {self.csv_path}") from error

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
    file = DataSet(csv_name="Social_Network_Ads.csv", train_ratio=0.6,
                   validation_ratio=0.2, test_ratio=0.2, seed=42)
