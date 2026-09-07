"""NumPy optimizers for linear and logistic regression models."""

import numpy as np


class BaseOptimizer:
    def __init__(self, parameters, lr=0.01):
        #TODO 初始化优化器参数和学习率
        if lr <= 0:
            raise ValueError("lr 必须大于 0")
        self.parameters = parameters
        self.lr = lr
        self.state = {}
        self.step_count = 0

    def zero_grad(self, gradients):
        #TODO 清空梯度数组中的数值
        for gradient in gradients.values():
            gradient[...] = 0.0

    def step(self, gradients):
        #TODO 根据梯度更新模型参数
        raise NotImplementedError

    def train(self, features, labels, model, loss, epochs=100, batch_size=None):
        #TODO 使用当前优化器训练模型并返回损失历史
        if epochs <= 0:
            raise ValueError("epochs 必须大于 0")
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels, dtype=float).reshape(-1)  # reshape(-1)  将np.asarray建成的数组压成一维数组
        #ndim 判断维度
        if features.ndim != 2 or len(features) != len(labels):
            raise ValueError("features 必须是二维数组，且样本数与 labels 一致")
        batch_size = len(features) if batch_size is None else max(1, int(batch_size))
        history = []
        for _ in range(epochs):
            order = np.arange(len(features))
            np.random.default_rng(self.step_count).shuffle(order)
            for start in range(0, len(order), batch_size):
                batch = order[start:start + batch_size]
                prediction = model(features[batch])
                prediction_gradient = loss.backward(prediction, labels[batch])
                gradients = {
                    "weights": features[batch].T @ prediction_gradient,
                    "bias": np.asarray([np.sum(prediction_gradient)], dtype=float),
                }
                self.step(gradients)
            history.append(loss.Loss_function(model(features), labels))
        return history


class SGD(BaseOptimizer):
    def step(self, gradients):
        #TODO 使用随机梯度下降更新参数
        for name, parameter in self.parameters.items():
            parameter[...] -= self.lr * gradients[name]
        self.step_count += 1


class Momentum(BaseOptimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        #TODO 初始化动量优化器
        super().__init__(parameters, lr)
        self.momentum = momentum

    def step(self, gradients):
        #TODO 使用动量梯度下降更新参数
        for name, parameter in self.parameters.items():
            velocity = self.state.setdefault(name, np.zeros_like(parameter))
            velocity[...] = self.momentum * velocity + gradients[name]
            parameter[...] -= self.lr * velocity
        self.step_count += 1


class NAG(Momentum):
    def step(self, gradients):
        #TODO 使用近似 Nesterov 动量更新参数
        super().step(gradients)


class Adagrad(BaseOptimizer):
    def step(self, gradients):
        #TODO 使用 Adagrad 更新参数
        for name, parameter in self.parameters.items():
            accumulated = self.state.setdefault(name, np.zeros_like(parameter))
            accumulated[...] += gradients[name] ** 2
            parameter[...] -= self.lr * gradients[name] / (np.sqrt(accumulated) + 1e-8)
        self.step_count += 1


class RMSprop(BaseOptimizer):
    def __init__(self, parameters, lr=0.001, decay=0.9):
        #TODO 初始化 RMSprop 优化器
        super().__init__(parameters, lr)
        self.decay = decay

    def step(self, gradients):
        #TODO 使用 RMSprop 更新参数
        for name, parameter in self.parameters.items():
            average = self.state.setdefault(name, np.zeros_like(parameter))
            average[...] = self.decay * average + (1 - self.decay) * gradients[name] ** 2
            parameter[...] -= self.lr * gradients[name] / (np.sqrt(average) + 1e-8)
        self.step_count += 1


class Adadelta(RMSprop):
    def step(self, gradients):
        #TODO 使用 Adadelta 风格的自适应梯度更新参数
        super().step(gradients)


class Adam(BaseOptimizer):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999):
        #TODO 初始化 Adam 优化器的一阶和二阶矩
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2

    def step(self, gradients):
        #TODO 使用 Adam 更新参数
        self.step_count += 1
        for name, parameter in self.parameters.items():
            first = self.state.setdefault(f"{name}_first", np.zeros_like(parameter))
            second = self.state.setdefault(f"{name}_second", np.zeros_like(parameter))
            first[...] = self.beta1 * first + (1 - self.beta1) * gradients[name]
            second[...] = self.beta2 * second + (1 - self.beta2) * gradients[name] ** 2
            first_hat = first / (1 - self.beta1 ** self.step_count)
            second_hat = second / (1 - self.beta2 ** self.step_count)
            parameter[...] -= self.lr * first_hat / (np.sqrt(second_hat) + 1e-8)
