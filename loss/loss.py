"""Loss functions used by the NumPy training loop."""
import numpy as np

class BaseLoss:
    def __init__(self):
        #TODO 初始化损失函数基类
        pass

    def Loss_function(self, prediction, target):
        #TODO 定义损失函数公式
        raise NotImplementedError

    def backward(self, prediction, target):
        #TODO 求损失函数对预测值的梯度
        raise NotImplementedError

    def __call__(self, prediction, target):
        #TODO 让损失函数可以像函数一样直接调用求损失值
        return self.Loss_function(prediction, target)


class MSELoss(BaseLoss):
    def Loss_function(self, prediction, target):
        #TODO 计算线性回归的均方误差损失
        prediction, target = self._validate(prediction, target)
        return float(np.mean((prediction - target) ** 2))

    def backward(self, prediction, target):
        #TODO 计算均方误差对预测值的梯度
        prediction, target = self._validate(prediction, target)
        return 2.0 * (prediction - target) / prediction.size

    @staticmethod
    def _validate(prediction, target):
        #TODO 将预测值和标签转换为可计算的同形状数组
        prediction = np.asarray(prediction, dtype=float).reshape(-1)
        target = np.asarray(target, dtype=float).reshape(-1)
        if prediction.shape != target.shape or prediction.size == 0:
            raise ValueError("prediction 和 target 必须是相同的非空形状")
        return prediction, target


class BCELoss(BaseLoss):
    def Loss_function(self, prediction, target):
        #TODO 计算逻辑回归的二元交叉熵损失
        prediction, target = self._validate(prediction, target)
        prediction = np.clip(prediction, 1e-12, 1.0 - 1e-12)
        return float(-np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction)))

    def backward(self, prediction, target):
        #TODO 交叉熵求导
        #交叉熵 L = -mean(y*log(p) + (1-y)*log(1-p))，对预测值求导并经过 sigmoid 化简后，
        #最终梯度就是交叉熵求导的经典结果 (p - y) / N
        prediction, target = self._validate(prediction, target)
        return (prediction - target) / prediction.size

    @staticmethod
    def _validate(prediction, target):
        #TODO 校验预测概率和二分类标签
        prediction = np.asarray(prediction, dtype=float).reshape(-1)
        target = np.asarray(target, dtype=float).reshape(-1)
        if prediction.shape != target.shape or prediction.size == 0:
            raise ValueError("prediction 和 target 必须是相同的非空形状")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError("BCELoss 的 prediction 和 target 不能包含 NaN 或 Inf")
        target_is_binary = np.isclose(target, 0.0, atol=1e-7) | np.isclose(target, 1.0, atol=1e-7)
        if not target_is_binary.all():
            raise ValueError("BCELoss 的 target 必须是 0 或 1")
        target = np.rint(target)
        if np.any((prediction < -1e-7) | (prediction > 1 + 1e-7)):
            raise ValueError("BCELoss 的 prediction 必须位于 [0, 1]")
        #cilp 把数组里的元素强制控制在设定的两个值之间  小于最小值的会改成最小值，大于最大值的会改成最大值
        prediction = np.clip(prediction, 1e-12, 1.0 - 1e-12)
        return prediction, target


LinearRegressionLoss = MSELoss
LogisticRegressionLoss = BCELoss
