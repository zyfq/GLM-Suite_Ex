"""Simple NumPy models with a PyTorch-like module interface."""

import json
from pathlib import Path

import numpy as np


class BaseModel:
    def __init__(self, input_dim):
        #TODO 初始化模型参数和输入维度
        if input_dim <= 0:
            raise ValueError("input_dim 必须大于 0")
        self.input_dim = input_dim
        self.weights = np.zeros(input_dim, dtype=float)
        self.bias = np.zeros(1, dtype=float)
        self.normalization_params = None

    def forward(self, features):
        #TODO 根据输入特征计算模型输出
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError("输入特征必须是二维数组，且列数与 input_dim 一致")
        return features @ self.weights + self.bias

    def __call__(self, features):
        #TODO 提供类似 PyTorch Module 的可调用模型接口
        return self.forward(features)

    def parameters(self):
        #TODO 返回供优化器更新的模型参数
        return {"weights": self.weights, "bias": self.bias}

    def save_weights(self, path, normalization_params=None):
        #TODO 将模型权重和归一化参数以文本形式保存到代码目录之外的文件
        path = Path(path)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            path = project_root / "runs" / "weights" / path
        if path.resolve().is_relative_to(Path(__file__).resolve().parent):
            raise ValueError("权重文件不能保存在代码文件所在的目录中")
        path = path.with_suffix(".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        params = normalization_params if normalization_params is not None else self.normalization_params

        content = {
            "model_type": type(self).__name__,
            "weights": {f"weight_{index + 1}": float(value) for index, value in enumerate(self.weights)},
            "bias": float(np.atleast_1d(self.bias)[0]),
        }
        if params is not None:
            content["normalization"] = {
                "feature_minimum": {f"feature_{index + 1}": float(value) for index, value in enumerate(params["feature_minimum"])},
                "feature_maximum": {f"feature_{index + 1}": float(value) for index, value in enumerate(params["feature_maximum"])},
            }
        path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load_weights(self, path):
        #TODO 从 JSON 权重文件读取模型参数和归一化参数
        content = json.loads(Path(path).read_text(encoding="utf-8"))
        if "weights" not in content or "bias" not in content:
            raise ValueError("权重文件缺少 weights 或 bias 部分")

        weights = np.asarray(list(content["weights"].values()), dtype=float)
        if weights.shape != self.weights.shape:
            raise ValueError("权重文件的维度与当前模型不一致")
        saved_bias = float(content["bias"])

        self.weights[...] = weights
        self.bias[...] = saved_bias
        normalization = content.get("normalization")
        if normalization is not None:
            minimum = np.asarray(list(normalization["feature_minimum"].values()), dtype=float)
            maximum = np.asarray(list(normalization["feature_maximum"].values()), dtype=float)
            if minimum.shape != self.weights.shape or maximum.shape != self.weights.shape:
                raise ValueError("权重文件中的归一化参数维度与模型不一致")
            self.normalization_params = {
                "feature_minimum": minimum,
                "feature_maximum": maximum,
            }
        else:
            self.normalization_params = None
        return self

    def predict_new_data(self, raw_features, weights_path=None):
        #TODO 读取权重文件并对原始新数据完成归一化和预测
        if weights_path is not None:
            self.load_weights(weights_path)
        if self.normalization_params is None:
            raise ValueError("权重文件中缺少归一化参数，无法处理原始新数据")
        minimum = self.normalization_params["feature_minimum"]
        maximum = self.normalization_params["feature_maximum"]
        raw_features = np.asarray(raw_features, dtype=float)
        if raw_features.ndim == 1:
            raw_features = raw_features.reshape(1, -1)
        if raw_features.shape[1] != self.input_dim:
            raise ValueError("新数据的特征列数与模型输入维度不一致")
        if not np.isfinite(raw_features).all():
            raise ValueError("新数据中包含缺失值或非数字数据")
        scale = np.where(maximum == minimum, 1.0, maximum - minimum)
        normalized = (raw_features - minimum) / scale
        return self.forward(normalized)


class LinearRegression(BaseModel):
    def forward(self, features):
        #TODO 计算线性回归预测值
        return super().forward(features)


class LogisticRegression(BaseModel):
    def forward(self, features):
        #TODO 计算逻辑回归概率预测值
        logits = super().forward(features)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))

    def predict(self, features, weights_path=None, threshold=0.5):
        #TODO 根据阈值将逻辑回归概率转换为分类结果
        if not 0 < threshold < 1:
            raise ValueError("threshold 必须位于 0 和 1 之间")
        probabilities = self.predict_proba(features, weights_path)
        return (probabilities >= threshold).astype(int)

    def predict_proba(self, features, weights_path=None):
        #TODO 返回逻辑回归概率预测值
        if weights_path is not None:
            self.load_weights(weights_path)
        return self.forward(features)

    def predict_new_data(self, raw_features, weights_path=None, threshold=0.5):
        #TODO 读取权重文件并对原始新数据完成归一化和分类预测
        if not 0 < threshold < 1:
            raise ValueError("threshold 必须位于 0 和 1 之间")
        probabilities = self.predict_new_proba(raw_features, weights_path)
        return (probabilities >= threshold).astype(int)

    def predict_new_proba(self, raw_features, weights_path=None):
        #TODO 读取权重文件并对原始新数据完成归一化和概率预测
        return BaseModel.predict_new_data(self, raw_features, weights_path)


LinearModel = LinearRegression
LogisticModel = LogisticRegression
