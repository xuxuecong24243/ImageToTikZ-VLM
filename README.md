# ImageToTikZ-VLM

将技术示意图和流程图转换为**可编辑的 LaTeX/TikZ**，并在编译依赖可用时生成 PDF 和 PNG 预览。

当前稳定基线为 **v0.5 Local**：使用本地 OCR 和 OpenCV 提取文字、图形、颜色与连接，由 Python 确定性构建 IR，经校验和 Scene 转换后生成 TikZ。输出包含独立的文字、节点和路径，不将原图嵌入 PDF 代替转换。

识别可能缺失或出错，**不保证像素级还原或语义完整**。复杂图片通常返回 `partial`，需要结合报告与原图检查。

## 环境要求

- 已验证：Windows、CPython **3.9.10**、CPU；其他 Python/操作系统组合尚未完成同等验收。
- 主环境：Pillow、PyMuPDF、NumPy、OpenCV，版本见 `requirements.txt` 和 `requirements-geometry.txt`。
- 独立 OCR 环境：RapidOCR ONNX，版本见 `requirements-ocr.txt`。不需要 PyTorch、CUDA 或 PaddlePaddle。
- 外部编译器：**XeLaTeX**，可使用 TeX Live 或 MiKTeX。需要 `standalone`、`ctex`、`amsmath`、`amssymb`、TikZ、Fandol 中文字体和 Latin Modern 字体。已在 TeX Live 2024 上实际编译。
- PNG 由 PyMuPDF 从 PDF 生成，无需额外安装 Poppler。

已验证的 OCR NumPy/OpenCV 版本与主环境不同。不要把 `requirements-ocr.txt` 安装到主环境，也不要为本项目升级其他项目的深度学习依赖。

## 安装

在项目根目录的 PowerShell 终端执行。先用 `python --version` 确认解释器；已有环境无需重复创建。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-geometry.txt

python -m venv .venv-ocr
.\.venv-ocr\Scripts\python.exe -m pip install -r requirements-ocr.txt

xelatex --version
.\.venv\Scripts\python.exe convert.py --help
```

安装依赖需要网络；RapidOCR 固定版本安装包包含所需 ONNX 模型。**Local 推理不联网、不读取 API 配置、不下载替代模型**。OCR 环境或模型缺失时明确失败，不回退到 VLM。

XeLaTeX 需在 `PATH` 中，也可追加 `--xelatex "实际安装路径\xelatex.exe"`。TeX 包和字体由 TeX 发行版安装，程序不会自动安装。以下离线自检验证中文、公式和箭头编译，手写自检图不代表图片识别：

```powershell
.\.venv\Scripts\python.exe main.py --smoke-test
```

## 快速开始

准备一张 PNG/JPG，放到 `inputs/example.png`，或直接使用自己的文件路径。**公开仓库不附带开发期原始图片或历史输出**；示例路径必须实际存在。中文或空格路径请加引号。

```powershell
.\.venv\Scripts\python.exe convert.py --input "inputs/example.png" --output "outputs/result" --backend local
```

如果本地已有 DataLoader 验收图 `inputs/01_figure.png`，可运行：

```powershell
.\.venv\Scripts\python.exe convert.py --input "inputs/01_figure.png" --output "outputs/dataloader_result" --backend local
```

输出目录必须不存在或为空，已有结果不会被覆盖。Local 不需要 `.env`、API Key、`--allow-api`，也不需要手工生成 IR、编辑 Scene 或分步执行脚本。

| 参数 | 用途 |
|---|---|
| `--ocr-python "路径\python.exe"` | 指定 OCR 解释器，默认使用项目 `.venv-ocr` |
| `--ocr-timeout 300` | OCR 工作进程超时，允许 1–900 秒 |
| `--xelatex "路径\xelatex.exe"` | 指定 XeLaTeX |
| `--compile-timeout 60` | 编译超时 |
| `--no-compile` | 仅生成 TeX，同时跳过字体测量和 PDF/PNG 预览 |

## 输出与状态

```text
outputs/result/
├── generated.tex
├── generated.tikz
├── generated.pdf          # 编译成功时
├── generated.png          # PDF 渲染成功时
├── comparison.png         # 原图与预览对照
├── report.json
└── ...                    # OCR、几何、IR、Scene、诊断与日志
```

| 文件 | 内容 |
|---|---|
| `generated.tex` | 完整可编辑 LaTeX 文档 |
| `generated.tikz` | 独立 TikZ 片段 |
| `generated.pdf` / `generated.png` | 实际编译和渲染的预览 |
| `report.json` | 总状态、输入哈希、请求次数、阶段结果、错误和文件路径 |
| `ocr_raw.json` / `geometry_raw.json` | 原 OCR 内容、置信度和像素几何 |
| `ir_generated.json` / `scene_generated.json` | 确定性中间结构和可绘制场景 |
| `small_shape_detection.json` / `connection_mapping.json` | 圆内文字来源、端点及未确认关系 |
| `pipeline.log`、`ocr_worker.log`、编译日志 | 详细故障定位信息 |

- `success` / **退出码 0**：转换及核心产物生成完成，不等于视觉或语义完全正确。
- `partial` / **退出码 2**：已有有效 TeX，但识别/转换不完整，或预览未能生成。当前 Local 识别通常保留此状态。
- `failed` / **退出码 1**：无法生成有效 TeX，查看具体失败阶段。

参数语法错误也可能返回 argparse 的退出码 2，应同时检查终端提示和报告。编译失败时保留 TeX 和日志；生成 PDF 不能单独证明内容完整。

## 模式与 API 配置

| 模式 | 状态 |
|---|---|
| `local` | **主要模式**，显式传入 `--backend local` |
| `vlm` | 可选图片模型模式，需要自己的配置和显式费用授权 |
| `hybrid` | `convert.py` 仅保留参数位置并明确拒绝执行；独立实验未证明视觉收益，不是正式功能 |

为兼容旧命令，省略 `--backend` 时仍选择 `vlm`；**这不会自动授权请求**。没有 `--allow-api` 时停止，不读取真实 API 配置或发送请求。

仅使用 VLM 时，在本地编辑器中将 `.env.example` 复制为 `.env`，填写 `API_BASE_URL`、`API_KEY`、`MODEL_NAME`。勿覆盖已有配置、提交真实密钥或把密钥放入命令历史。进程环境优先于项目根 `.env`；选择服务商明确支持图片输入的模型，程序不会自动切换模型。

```powershell
.\.venv\Scripts\python.exe convert.py --input "inputs/example.png" --output "outputs/vlm_result" --backend vlm --allow-api --max-api-requests 1
```

这会将图片发送给已配置服务并可能产生费用。统一入口最多一次请求，失败也计数，不自动重试或另发修复请求。Local 不接受 `--allow-api` 或 `--offline-ir`，避免混用流程。

旧入口、两阶段模式和离线 IR 回放见 [兼容 CLI 参考](docs/legacy_usage.md)；Hybrid 与历史实验保留范围见 [实验说明](docs/experiments.md)。

## 已知限制

- OCR 可能识别错字、漏掉小字；代码下划线、括号和大小写需要核对。置信度不是准确率。
- 复杂公式、上下标和生僻字符可能识别或排版错误，不会补写缺失数学含义。
- 三维张量、复杂科学插图、曲线、重复虚线框及复杂残差连接的还原不足。
- 支持部分小圆文字及有实心头部的水平、垂直、简单折线箭头；开口头、边框附着、遮挡和分支仍可能遗漏。无法确认的端点保留诊断，不猜连接。
- 实测路径与规范锚点存在间隙时可能报告 `connection_path_anchor_mismatch`；IR 合法不代表所有图元均能转换。
- 字体测量依赖完整可转换的结构和可用字体，失败时保留原排版。颜色、圆角、图层和布局只能近似重建。
- 不保证 100% 视觉一致；有限验收图片上的成功不能推广到所有陌生图片。

## 常见问题

| 提示/现象 | 处理方式 |
|---|---|
| `Local OCR runtime is missing` | 创建 `.venv-ocr` 并安装 OCR 依赖，或检查 `--ocr-python` |
| OCR 模型缺失或超时 | 查看 `ocr_worker.log`，核对独立环境安装；不会自动联网补模型 |
| 找不到 XeLaTeX / 缺包或字体 | 安装对应 TeX 组件，检查 `PATH` 或 `--xelatex`；可先用 `--no-compile` 保留源文件 |
| 输出目录不为空 | 改用新目录，保留已有验收结果 |
| 图片不存在或 EXIF 旋转未归一化 | 使用实际单帧 PNG/JPG，先旋转保存为正确方向 |
| 有 PDF 但为 `partial` | 查看报告、转换诊断并对照原图，检查识别与表达限制 |

## 测试与代码导航

安装主环境和几何依赖后，在项目根目录执行完整测试集：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

测试使用标准库 `unittest`，真实配置与网络由测试隔离器阻断。Mock 测试验证协议和错误处理，不能替代真实 OCR/XeLaTeX 验收；完整转换应使用自己的图片单独验证。

公开仓库包含经过机器路径脱敏的 `tests/fixtures/historical/`，可在没有历史 `outputs/` 的检出目录运行全部 56 个 Scene 校验和 49 个 TikZ 哈希回归。只有私有原始档案的字节完整性检查会在档案缺失时明确跳过；这不会跳过公开兼容性验证。测试数据是历史参考，不代表新的自动识别结果。

```text
convert.py                     统一入口
local_ocr.py                    独立 CPU OCR 工作进程
geometry_analyzer.py            通用 OpenCV 几何分析
local_geometry.py / local_fusion.py
local_small_shapes.py           小圆与内部文字
local_arrow_detection.py / local_connection_mapping.py
visual_style_extractor.py / local_visual_layers.py
text_layout_analysis.py / text_measurement.py / local_text_layout.py
local_pipeline.py              Local 流程编排
structured_ir/                 IR 契约、校验、转换与诊断
structured_tikz/               Scene 契约与 TikZ 生成
auto_compile.py / image_compare.py
vlm_client.py                  可选 VLM 接口与授权门控
tests/                         离线测试与回归
examples/                      结构示例，不代表自动识别
experiments/                   独立实验，不进入 Local 主流程
docs/                          兼容入口与实验说明
inputs/ / outputs/             用户本地数据与产物
```

编译使用 `-no-shell-escape` 和有限资源约束；静态检查不构成完整 TeX 安全沙箱。公开服务部署还需要文件系统、网络和权限隔离。
