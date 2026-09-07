# GLM-Suite_Ex

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GLM-Suite_Ex** (Generalized Linear Model Suite - Extended) 是一个从零实现、NumPy 驱动的广义线性模型训练框架。本项目以清晰模块化的设计，同时支持**线性回归**（连续值预测）与**逻辑回归**（二分类），并集成了 **7 种梯度下降优化算法**，便于深入对比不同优化器的效果。

> 📌 **核心定位**：机器学习算法学习与实验工具，适合教学演示、算法对比研究。

---

## ✨ 特性

- **双模型支持**：线性回归与逻辑回归，统一 `BaseModel` 接口[reference:6]
- **7 种优化器**：SGD、Momentum、NAG、Adagrad、RMSprop 等[reference:7]
- **模块化设计**：模型、优化器、损失函数、数据处理、可视化完全解耦
- **训练管理**：支持批量训练、数据打乱、损失历史记录[reference:8]
- **模型持久化**：权重保存与加载（JSON 格式），支持迁移预测[reference:9]
- **数据标准化**：内置特征归一化与反归一化[reference:10]
- **可视化工具**：损失曲线、决策边界绘制[reference:11]

---

## 🛠 技术栈

| 组件 | 说明 |
| :--- | :--- |
| **语言** | Python 3.8+ |
| **核心库** | NumPy |
| **可视化** | Matplotlib |
| **数据处理** | Pandas (可选) |

---

## 📁 项目结构
