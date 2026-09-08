from pathlib import Path

import numpy as np

from dataoperation.data import DataSet
from loss.loss import BCELoss, MSELoss
from model.model import LinearRegression, LogisticRegression
from optimizer.optimizer import SGD, Momentum, NAG, Adagrad, RMSprop, Adadelta, Adam
from picturedraw.draw import Plotter


def main(task="linear"):
    """根据 task 训练线性回归或逻辑回归，并保存对应图表。"""
    project_root = Path(__file__).resolve().parent
    task_configs = {
        "linear": {
            "dataset": "LineRegression/Linear Regression - Sheet1.csv",
            "loss": MSELoss,
            "model": LinearRegression,
            "display_name": "线性回归",
            "new_samples": np.array([[10.0], [50.0], [100.0]]),
        },
        "logistic": {
            "dataset": "Social_Network_Ads.csv",
            "loss": BCELoss,
            "model": LogisticRegression,
            "display_name": "逻辑回归",
            "new_samples": np.array([[30.0, 60000.0], [45.0, 100000.0], [60.0, 20000.0]]),
        },
    }
    if task not in task_configs:
        raise ValueError("task 必须是 'linear' 或 'logistic'")
    config = task_configs[task]

    dataset = DataSet(csv_name=config["dataset"], train_ratio=0.6,
                      validation_ratio=0.2, test_ratio=0.2, seed=40)
    loss = config["loss"]()
    model_class = config["model"]
    task_name = config["display_name"]
    train_features, train_labels = dataset.train_data
    val_features, val_labels = dataset.validation_data
    test_features, test_labels = dataset.test_data

    # 七种优化算法的配置，各优化器使用独立模型进行公平比较
    optimizer_configs = {
        "随机梯度下降": (SGD, {"lr": 0.05}),
        "动量梯度下降": (Momentum, {"lr": 0.02, "momentum": 0.9}),
        "NAG": (NAG, {"lr": 0.02, "momentum": 0.9}),
        "Adagrad": (Adagrad, {"lr": 0.1}),
        "RMSprop": (RMSprop, {"lr": 0.01, "decay": 0.9}),
        "Adadelta": (Adadelta, {"lr": 0.01, "decay": 0.9}),
        "Adam": (Adam, {"lr": 0.01, "beta1": 0.9, "beta2": 0.999}),
    }

    epochs = 300
    batch_size = 32
    plotter = Plotter(save_dir=project_root / "runs" / "plots")
    validation_histories = {}
    results = {}

    for optimizer_name, (optimizer_class, optimizer_kwargs) in optimizer_configs.items():
        model = model_class(input_dim=train_features.shape[1])
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        train_history = []
        validation_history = []

        print(f"\n开始使用{optimizer_name}训练{task_name}")
        for epoch in range(epochs):
            epoch_history = optimizer.train(
                train_features,
                train_labels,
                model,
                loss,
                epochs=1,
                batch_size=batch_size,
            )
            train_history.append(epoch_history[0])
            validation_history.append(loss(model(val_features), val_labels))

            if (epoch + 1) % 50 == 0:
                print(
                    f"第 {epoch + 1:03d}/{epochs} 轮 | "
                    f"训练损失: {train_history[-1]:.6f} | "
                    f"验证损失: {validation_history[-1]:.6f}"
                )

        test_predictions = model(test_features)
        test_loss = loss(test_predictions, test_labels)
        validation_histories[optimizer_name] = validation_history
        results[optimizer_name] = {
            "model": model,
            "test_loss": test_loss,
            "validation_loss": validation_history[-1],
        }

        safe_name = optimizer_class.__name__.lower()
        plotter.plot_convergence(
            train_loss=train_history,
            val_loss=validation_history,
            filename=f"{task}/optimizers/{safe_name}/convergence",
        )
        if task == "linear":
            plotter.plot_linear_regression(
                features=test_features,
                labels=test_labels,
                model=model,
                feature_index=0,
                filename=f"linear/optimizers/{safe_name}/model_function",
            )
        else:
            plotter.plot_logistic_regression(
                features=test_features,
                labels=test_labels,
                model=model,
                feature_indices=(0, 1),
                filename=f"logistic/optimizers/{safe_name}/test_result",
            )
        print(f"{optimizer_name} 测试集损失: {test_loss:.6f}")

    comparison_path = plotter.plot_optimizer_comparison(
        histories=validation_histories,
        filename=f"{task}/optimizers/comparison",
    )
    best_name = min(results, key=lambda name: results[name]["validation_loss"])
    best_result = results[best_name]
    best_model = best_result["model"]

    print(f"\n七种优化算法的{task_name}测试结果：")
    for optimizer_name, result in results.items():
        print(
            f"{optimizer_name:<8} | 测试损失: {result['test_loss']:.6f} | "
            f"最终验证损失: {result['validation_loss']:.6f}"
        )
    print(
        f"\n验证集表现最佳的优化器：{best_name}，"
        f"测试损失：{best_result['test_loss']:.6f}"
    )
    print(f"七种优化器对比图已保存到：{comparison_path}")

    weights_path = best_model.save_weights(
        f"{task}_model.txt",
        normalization_params={
            "feature_minimum": dataset.feature_minimum,
            "feature_maximum": dataset.feature_maximum,
        },
    )
    new_samples = config["new_samples"]
    deployed_model = model_class(input_dim=train_features.shape[1])

    if task == "linear":
        new_predictions = deployed_model.predict_new_data(new_samples, weights_path)
        print(f"最佳线性回归模型权重已保存到：{weights_path}")
        for sample, prediction in zip(new_samples, new_predictions):
            print(f"新样本: X={sample[0]:.2f} | 线性回归预测值: {prediction:.4f}")
    else:
        new_probabilities = deployed_model.predict_new_proba(new_samples, weights_path)
        new_labels = deployed_model.predict_new_data(new_samples, weights_path)
        print(f"最佳逻辑回归模型权重已保存到：{weights_path}")
        for sample, probability, label in zip(new_samples, new_probabilities, new_labels):
            print(
                f"新样本: 年龄={sample[0]:.0f} 薪资={sample[1]:.0f} | "
                f"购买概率: {probability:.4f} | 预测结果: {label}"
            )


if __name__ == "__main__":
    main(task="linear")
