# 实验代码与发布范围

当前主要路径是 **v0.5 Local**，见 [快速开始](../README.md)。本地 OCR、几何、样式、排版、小圆及常见箭头能力已集成到 `convert.py --backend local`；实验目录不会自动进入这条路径。

## 保留原则

整理优先保持运行和测试依赖。具有通用功能或被测试引用的旧入口保留，不因名字包含 `experiment`、`repair` 或 `test` 删除。单图修复脚本、临时调试结果和历史报告主要位于 `outputs/`，本地保留供追溯，不随公开仓库发布。

原始输入图片、API 响应、完整运行目录、编译日志、模型文件、虚拟环境和缓存不属于发布内容。`examples/` 的结构示例是显式参考数据，不是模型自动识别结果。

## 独立 Hybrid 实验

保留：

- `experiments/hybrid_structure.py`：结构补充契约、严格解析和融合保护。
- `experiments/run_hybrid_experiment.py`：prepare / prompt / request / render，在线阶段需显式授权，持久化账本限制一次请求。
- 对应测试：ID、文字/坐标保护、冲突拒绝、原响应保存及请求预算。

这些代码仍有测试和参考价值，但**不是正式 Hybrid 产品模式**。`convert.py --backend hybrid` 当前明确报告未实现，不自动转入独立实验。

2026-11-23 的一次 Transformer 受限实验中，模型返回可解析的分组、连接和候选信息，但 Decoder 与输出区存在 ID 语义错配，并提出残差自连接。既定规则没有接纳新增连接，只保存一项训练卡片包含元数据；实际 Hybrid TikZ、TeX、PNG 与 v0.5 Local 逐字节相同。因此**没有证明视觉收益**，不能宣传已恢复复杂结构。

实验未改变 Local/OCR/OpenCV/Schema/生成器。曲线、虚线框、加号候选保持未验证，没有人工修模型输出或补画。真实响应、基线和报告只保留在开发环境的 `outputs/hybrid_experiment_transformer_20261123/`，公共仓库不要求包含这些数据。

实验默认参数引用历史输入和基线，公共检出不保证存在。运行前必须提供自己的真实输入、自动生成基线和新输出目录；不要把手工 Scene 或参考 IR 当作自动识别。安装、自检和发布验证不需要运行在线请求。

## 可选 VLM 能力

`vlm_client.py` 及 `convert.py --backend vlm` 保留显式授权的图片模型接口；`main.py`、`auto_pipeline.py` 与两阶段入口见 [兼容 CLI 参考](legacy_usage.md)。

没有 API 授权不发送真实请求。Local 不加载 API 配置、不调用 VLM，也不在识别失败后自动切换模型。保留接口不意味着所有服务、模型或陌生图片已验收。
