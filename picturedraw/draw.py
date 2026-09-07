"""训练过程、线性回归与逻辑回归结果的可视化工具。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """以类似 PyTorch Module 的调用方式生成并保存训练图表。

    默认输出到项目根目录的 ``runs/plots``，不会与本代码文件保存在同一目录。
    所有绘图方法均返回生成图片的 :class:`pathlib.Path`。
    """

    SUPPORTED_TASKS = {"convergence", "linear", "logistic"}

    def __init__(self, save_dir=None, image_format="png", dpi=150):
        project_root = Path(__file__).resolve().parent.parent
        self.save_dir = Path(save_dir) if save_dir else project_root / "runs" / "plots"
        if self.save_dir.resolve() == Path(__file__).resolve().parent:
            raise ValueError("图片目录不能与绘图代码位于同一个目录")

        self.image_format = image_format.lstrip(".").lower()
        if self.image_format not in {"png", "jpg", "jpeg", "svg", "pdf"}:
            raise ValueError("image_format 必须是 png、jpg、jpeg、svg 或 pdf")
        if dpi <= 0:
            raise ValueError("dpi 必须大于 0")

        self.dpi = dpi
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._set_chinese_font()

    def __call__(self, task, **kwargs):
        """根据任务名称调用对应绘图方法，风格类似 PyTorch 的可调用模块。"""
        return self.forward(task, **kwargs)

    def forward(self, task, **kwargs):
        handlers = {
            "convergence": self.plot_convergence,
            "linear": self.plot_linear_regression,
            "logistic": self.plot_logistic_regression,
        }
        if task not in handlers:
            raise ValueError(f"task 必须是以下值之一: {sorted(self.SUPPORTED_TASKS)}")
        return handlers[task](**kwargs)

    @staticmethod
    def _set_chinese_font():
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
        ]
        plt.rcParams["axes.unicode_minus"] = False

    @staticmethod
    def _as_1d(values, name):
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"{name} 必须是非空且不含 NaN/Inf 的数组")
        return array

    def _output_path(self, filename):
        filename = Path(filename)
        if filename.is_absolute():
            raise ValueError("filename 只能是相对于 save_dir 的相对路径")
        if filename.suffix:
            filename = filename.with_suffix(f".{self.image_format}")
        else:
            filename = filename.parent / f"{filename.name}.{self.image_format}"
        output_path = self.save_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _save(self, figure, filename):
        output_path = self._output_path(filename)
        figure.tight_layout()
        figure.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(figure)
        return output_path

    def plot_convergence(self, train_loss, val_loss=None, filename="train/convergence"):
        """绘制训练集及可选验证集的损失收敛曲线。"""
        train_loss = self._as_1d(train_loss, "train_loss")
        epochs = np.arange(1, train_loss.size + 1)

        figure, axes = plt.subplots(figsize=(8, 5))
        axes.plot(epochs, train_loss, color="#2563eb", linewidth=2, label="训练损失")
        if val_loss is not None:
            val_loss = self._as_1d(val_loss, "val_loss")
            if val_loss.size != train_loss.size:
                raise ValueError("val_loss 与 train_loss 的长度必须相同")
            axes.plot(epochs, val_loss, color="#ef4444", linewidth=2, label="验证损失")
        axes.set(xlabel="训练轮次", ylabel="损失值", title="模型训练收敛曲线")
        axes.grid(alpha=0.25)
        axes.legend()
        return self._save(figure, filename)

    def plot_optimizer_comparison(self, histories, filename="optimizers/comparison"):
        """在同一张图中比较多种优化器的验证损失收敛过程。"""
        if not histories:
            raise ValueError("histories 不能为空")

        figure, axes = plt.subplots(figsize=(10, 6))
        for optimizer_name, losses in histories.items():
            losses = self._as_1d(losses, f"{optimizer_name} losses")
            epochs = np.arange(1, losses.size + 1)
            axes.plot(epochs, losses, linewidth=1.8, label=optimizer_name)

        axes.set(xlabel="训练轮次", ylabel="验证损失", title="七种优化算法收敛效果对比")
        axes.grid(alpha=0.25)
        axes.legend(ncol=2)
        return self._save(figure, filename)

    def plot_linear_regression(self, features, labels, model=None, predictions=None,
                               feature_index=0, filename="linear/model_fit"):
        """绘制线性回归样本散点图和模型拟合曲线。"""
        features = np.asarray(features, dtype=float)
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        labels = self._as_1d(labels, "labels")
        if features.ndim != 2 or len(features) != len(labels):
            raise ValueError("features 必须是二维数组，且样本数与 labels 相同")
        if not 0 <= feature_index < features.shape[1]:
            raise ValueError("feature_index 超出了特征列范围")

        predictions = self._get_predictions(features, model, predictions)
        order = np.argsort(features[:, feature_index])
        x_axis = features[order, feature_index]

        figure, axes = plt.subplots(figsize=(8, 5))
        axes.scatter(features[:, feature_index], labels, alpha=0.7,
                     color="#2563eb", label="真实样本")
        axes.plot(x_axis, predictions[order], color="#ef4444",
                  linewidth=2, label="模型预测")
        axes.set(xlabel=f"特征 {feature_index + 1}", ylabel="目标值",
                 title="线性回归模型拟合结果")
        axes.grid(alpha=0.25)
        axes.legend()
        return self._save(figure, filename)

    def plot_logistic_regression(self, features, labels, model=None, probabilities=None,
                                 feature_indices=(0, 1), filename="logistic/overview"):
        """绘制概率直方图、模型概率函数图和分类样本散点图。"""
        features = np.asarray(features, dtype=float)
        labels = self._as_1d(labels, "labels")
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        if features.ndim != 2 or len(features) != len(labels):
            raise ValueError("features 必须是二维数组，且样本数与 labels 相同")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("逻辑回归标签必须为 0 或 1")

        probabilities = self._get_predictions(features, model, probabilities)
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("逻辑回归预测概率必须位于 [0, 1]")

        figure, axes = plt.subplots(1, 3, figsize=(17, 5))
        self._plot_probability_histogram(axes[0], probabilities, labels)
        self._plot_probability_function(axes[1], features, probabilities, feature_indices[0])
        self._plot_class_scatter(axes[2], features, labels, probabilities, feature_indices)
        return self._save(figure, filename)

    def _get_predictions(self, features, model, predictions):
        if predictions is None:
            if model is None or not callable(model):
                raise ValueError("model 和 predictions 至少需要提供一个")
            predictions = model(features)
        predictions = self._as_1d(predictions, "predictions")
        if predictions.size != len(features):
            raise ValueError("预测结果数量必须与样本数相同")
        return predictions

    @staticmethod
    def _plot_probability_histogram(axes, probabilities, labels):
        bins = np.linspace(0, 1, 16)
        axes.hist(probabilities[labels == 0], bins=bins, alpha=0.7,
                  color="#2563eb", label="类别 0")
        axes.hist(probabilities[labels == 1], bins=bins, alpha=0.7,
                  color="#ef4444", label="类别 1")
        axes.set(xlabel="预测概率", ylabel="样本数量", title="预测概率分布")
        axes.legend()

    @staticmethod
    def _plot_probability_function(axes, features, probabilities, feature_index):
        if not 0 <= feature_index < features.shape[1]:
            raise ValueError("feature_indices 中的索引超出了特征列范围")
        order = np.argsort(features[:, feature_index])
        axes.scatter(features[:, feature_index], probabilities, s=18, alpha=0.55,
                     color="#2563eb", label="预测概率")
        axes.plot(features[order, feature_index], probabilities[order],
                  color="#ef4444", alpha=0.8, label="模型概率函数")
        axes.axhline(0.5, color="#111827", linestyle="--", label="分类阈值")
        axes.set(xlabel=f"特征 {feature_index + 1}", ylabel="属于类别 1 的概率",
                 title="逻辑回归模型函数")
        axes.legend()

    @staticmethod
    def _plot_class_scatter(axes, features, labels, probabilities, feature_indices):
        first, second = feature_indices
        if features.shape[1] == 1:
            first = second = 0
            y_axis = probabilities
            y_label = "属于类别 1 的概率"
        else:
            if not 0 <= first < features.shape[1] or not 0 <= second < features.shape[1]:
                raise ValueError("feature_indices 中的索引超出了特征列范围")
            y_axis = features[:, second]
            y_label = f"特征 {second + 1}"

        colors = np.where(labels == 1, "#ef4444", "#2563eb")
        axes.scatter(features[:, first], y_axis, c=colors, alpha=0.75)
        for class_value, color in ((0, "#2563eb"), (1, "#ef4444")):
            axes.scatter([], [], color=color, label=f"真实类别 {class_value}")
        axes.set(xlabel=f"特征 {first + 1}", ylabel=y_label, title="分类样本散点图")
        axes.legend()


Draw = Plotter

