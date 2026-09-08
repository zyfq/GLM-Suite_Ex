from pathlib import Path

import numpy as np

from dataoperation.data import DataSet
from loss.loss import BCELoss
from model.model import LogisticRegression
from optimizer.optimizer import SGD, Momentum, NAG, Adagrad, RMSprop, Adadelta, Adam
from picturedraw.draw import Plotter


def main():
    """使用七种优化器训练逻辑回归模型并比较训练效果。"""
    project_root = Path(__file__).resolve().parent
    #数据集统一从项目外的共享目录读取，可被多个项目共用
    dataset = DataSet(
        csv_name="Social_Network_Ads.csv",
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        seed=40,
    )
    #训练集 验证集和测试集的特征数据和标签数据
    train_features, train_labels = dataset.train_data
    val_features, val_labels = dataset.validation_data
    test_features, test_labels = dataset.test_data
    #利用字典封装各个算法的参数
    optimizer_configs = {
        "随机梯度下降": (SGD, {"lr": 0.1}),
        "动量梯度下降": (Momentum, {"lr": 0.05, "momentum": 0.9}),
        "NAG": (NAG, {"lr": 0.05, "momentum": 0.9}),
        "Adagrad": (Adagrad, {"lr": 0.1}),
        "RMSprop": (RMSprop, {"lr": 0.01, "decay": 0.9}),
        "Adadelta": (Adadelta, {"lr": 0.01, "decay": 0.9}),
        "Adam": (Adam, {"lr": 0.01, "beta1": 0.9, "beta2": 0.999}),
    }

    epochs = 300  #循环次数
    batch_size = 32  #批大小
    loss = BCELoss()  #逻辑回归损失函数类对象
    plotter = Plotter(save_dir=project_root / "runs" / "plots")   #图画类对象
    validation_histories = {}
    results = {}

    for optimizer_name, (optimizer_class, optimizer_kwargs) in optimizer_configs.items():
        model = LogisticRegression(input_dim=train_features.shape[1])
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        train_history = []
        validation_history = []

        print(f"\n开始训练：{optimizer_name}")
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

        test_probabilities = model.predict_proba(test_features)
        test_predictions = (test_probabilities >= 0.5).astype(int)
        test_accuracy = float(np.mean(test_predictions == test_labels))
        validation_histories[optimizer_name] = validation_history
        results[optimizer_name] = {
            "model": model,
            "accuracy": test_accuracy,
            "validation_loss": validation_history[-1],
        }

        safe_name = optimizer_class.__name__.lower()
        plotter.plot_convergence(
            train_loss=train_history,
            val_loss=validation_history,
            filename=f"optimizers/{safe_name}/convergence",
        )
        plotter.plot_logistic_regression(
            features=test_features,
            labels=test_labels,
            model=model,
            feature_indices=(0, 1),
            filename=f"optimizers/{safe_name}/test_result",
        )
        print(f"{optimizer_name} 测试集准确率: {test_accuracy:.2%}")

    comparison_path = plotter.plot_optimizer_comparison(
        histories=validation_histories,
        filename="optimizers/comparison",
    )
    best_name = min(results, key=lambda name: results[name]["validation_loss"])
    best_result = results[best_name]
    best_model = best_result["model"]

    print("\n七种优化算法测试结果：")
    for optimizer_name, result in results.items():
        print(
            f"{optimizer_name:<8} | 测试准确率: {result['accuracy']:.2%} | "
            f"最终验证损失: {result['validation_loss']:.6f}"
        )
    print(
        f"\n验证集表现最佳的优化器：{best_name}，"
        f"测试准确率：{best_result['accuracy']:.2%}"
    )
    print(f"七种优化器对比图已保存到：{comparison_path}")

    weights_path = best_model.save_weights(
        "logistic_model.txt",
        normalization_params={
            "feature_minimum": dataset.feature_minimum,
            "feature_maximum": dataset.feature_maximum,
        },
    )
    print(f"最佳模型权重与归一化参数已保存到：{weights_path}")

    deployed_model = LogisticRegression(input_dim=train_features.shape[1])
    new_samples = np.array(
        [
            [30.0, 60000.0],
            [45.0, 100000.0],
            [60.0, 20000.0],
        ]
    )
    new_probabilities = deployed_model.predict_new_proba(new_samples, weights_path)
    new_labels = deployed_model.predict_new_data(new_samples, weights_path)
    for sample, probability, label in zip(new_samples, new_probabilities, new_labels):
        print(
            f"新样本: 年龄={sample[0]:.0f} 薪资={sample[1]:.0f} | "
            f"购买概率: {probability:.4f} | 预测结果: {label}"
        )


if __name__ == "__main__":
    main()
