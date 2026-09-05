# Tennis Speed Estimator (YOLOv8)

该仓库包含使用 YOLOv8 检测网球并估算速度的分析与示例脚本。

内容：
- README.md：总体流程、训练与标定方法、实现建议、精度/进阶改进等（中文）。
- speed_estimator.py：示例测速脚本，包含检测、简单跟踪、像素->场地平面（单应）映射以及速度计算的最小可运行实现。
- LICENSE：MIT
- .gitignore：Python 常用忽略项

快速链接：
https://github.com/mathslingo/tennis-speed-estimator

说明：仓库由 Copilot 为用户 mathslingo 创建并初始化为 public。请按照 README 中的说明补全单应矩阵 H 或提供标定点，然后运行示例脚本。
