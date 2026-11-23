# 兼容 CLI 与数据接口参考

新用户先使用 [README](../README.md) 的 `convert.py --backend local`。以下入口为兼容保留，参数与请求预算各自独立；不要把一个入口的授权范围套用到另一个入口。示例从项目根目录执行，图片和 JSON 必须实际存在，输出使用新目录。

开发期逐轮日志和验收报告保留在本地 `outputs/`，不随公开仓库发布。本页保留可用操作说明，不要求用户持有这些历史产物。

## 统一入口离线 IR 回放

```powershell
.\.venv\Scripts\python.exe convert.py --input "inputs/example.png" --output "outputs/replay_result" --offline-ir "path/to/ir.json"
```

已有合法 IR 的画布必须匹配图片尺寸。回放不读 API 配置或调用模型，不与 `--allow-api` 组合，也不与 `--backend local` 组合。没有重新识别图片，因此不能将回放称为陌生图片自动识别。

## 原始 TikZ 模型入口：main.py

```powershell
.\.venv\Scripts\python.exe main.py --check
.\.venv\Scripts\python.exe main.py --smoke-test
```

`--check` 检查环境与配置，不请求 API；`--smoke-test` 是离线手写编译自检，不是图片识别。中文模板使用 Fandol，依赖 `standalone`、`ctex`、TikZ、`amsmath`、`amssymb`。

原在线入口让模型直接返回 TikZ，以下命令只允许一次生成：

```powershell
.\.venv\Scripts\python.exe main.py --image "inputs/example.png" --allow-api --max-repairs 0
```

**main.py 的 `--allow-api` 默认允许一次生成及最多两次编译修复**；只想请求一次必须加 `--max-repairs 0`。它不同于 `convert.py` 的一次请求上限。没有 HTTP 自动重试，不安全片段、API 错误或渲染错误不触发额外付费修复。

模型须输出单一 `tikzpicture` 片段及受支持的样式/宏，不能带完整文档或嵌入原图。Python 静态检查后包裹中文模板，编译失败保留日志。静态检查不是完整安全沙箱。

## 自动 Scene 入口：auto_pipeline.py

只准备原图与 Prompt，不请求或编译：

```powershell
.\.venv\Scripts\python.exe auto_pipeline.py --image "inputs/example.png" --dry-run
```

一次真实生成：

```powershell
.\.venv\Scripts\python.exe auto_pipeline.py --image "inputs/example.png" --allow-api --max-api-requests 1
```

默认 `--pipeline-mode single-stage`，模型输出 Scene，程序校验后确定性生成 TikZ。默认 `--geometry-mode assist`；用 `off` 关闭几何辅助。补充/视觉反馈不会自动启用。

| 参数 | 作用 |
|---|---|
| `--max-api-requests` | 总请求上限，默认 1，失败计数 |
| `--max-output-budget` | 累计预留输出 Token 上限，默认 65536 |
| `--max-total-usage-tokens` | 根据已返回 usage 阻止后续调用，默认 200000；不是请求前硬费用上限 |
| `--max-iterations` / `--max-local-rounds` | 默认 3 次迭代、2 轮有限本地修正 |
| `--stage-strategy adaptive --allow-completion` | 单独授权一次缺失字段补充，还须提高总请求预算 |
| `--allow-visual-refinement` | 单独授权一次双图反馈，还须提高总请求预算 |

补充和反馈共享请求、Token、迭代预算，单独 `--allow-api` 不授权第二次请求。JSON 错误、截断不会反复重试；非法原子 Patch 整体拒绝。有限本地修正不证明模型坐标正确。

`--offline-scene "path/to/scene.json"` 仅测试已提供场景，不与在线授权组合。`--reference "examples/gpt_embedding_scene.json"` 仅在结束后评估，不参与生成；示例不是自动识别数据。

## 两阶段 IR

```text
图片及区域候选 → VLM IR / 已保存 IR → IR Validator
→ 确定性 IR-to-Scene → Scene Validator → TikZ → XeLaTeX → PNG
```

```powershell
.\.venv\Scripts\python.exe auto_pipeline.py --pipeline-mode two-stage --image "inputs/example.png" --dry-run --output-root "outputs/ir_prepare"

.\.venv\Scripts\python.exe auto_pipeline.py --pipeline-mode two-stage --image "inputs/example.png" --offline-ir "path/to/ir.json" --geometry-mode off --output-root "outputs/ir_replay"
```

在线识别需要显式 `--allow-api --max-api-requests 1`；第二阶段在本地执行，不自动追加 VLM 修复。没有 `--dry-run`、`--offline-ir` 或 `--allow-api` 时以 `awaiting_ir` 停止，不用参考数据填充。

`--geometry-cache` 只接受与源图哈希匹配的几何缓存。`--no-compile` 用于 IR 校验/TeX 生成，不执行预览编译。离线原 IR、模型原文和诊断按新目录保存，不覆盖以前结果。

## 配置与输出预算

必要配置为 `API_BASE_URL`、`API_KEY`、`MODEL_NAME`。服务和模型必须支持 OpenAI-compatible 图片 Chat Completions。参考 `.env.example`，不把密钥放进文档或版本控制。

优先级为适用的 CLI > 进程环境 > 项目根 `.env` > 可选默认值。加载配置不授权请求。兼容 `API_BASE`，同一来源以 `API_BASE_URL` 为准；显式空环境变量清除文件同名设置。`.env` 按字面解析，不执行命令/插值，不搜索其他项目。

| 配置 | 对应参数/说明 |
|---|---|
| `API_MAX_OUTPUT_TOKENS` | `--max-tokens`；main.py 另有 `--max-output-tokens` 别名 |
| `API_TOKEN_LIMIT_PARAMETER` | main.py 的 `--token-limit-parameter`；仅使用服务明确支持的字段 |
| `API_MODEL_OUTPUT_LIMIT` | main.py 的 `--model-output-limit`；是输出限制而非上下文长度 |
| `API_THINKING_MODE` | main.py / auto_pipeline.py 的 `--thinking-mode` |
| `API_REASONING_EFFORT` | main.py / auto_pipeline.py 的 `--reasoning-effort` |

客户端仅对已内置的官方主机/精确模型组合设置特殊默认值；其他模型使用通用默认，不自动查询或切换服务。具体价格和能力以服务商当时说明为准。统一入口只暴露部分参数，使用各脚本 `--help` 查看实际选项。

`finish_reason=length`、拒绝、空响应或无效 JSON 会失败；增加预算不保证识别质量。诊断保存允许记录的模型名、预算、finish_reason 和 usage；未返回用量时不编造。main.py 的 `--save-api-diagnostics` 可保存允许留存的截断可见正文，但不会续写、编译截断内容或冒充成功结果。

## 稳定数据接口

- `structured_ir.schema.ir_schema()` / `structured_ir.validator.validate_ir()`：独立 IR 契约、坐标及引用约束。
- `structured_ir.converter.convert_ir()`：确定性返回 Scene 和诊断，无法表达的内容明确报告。
- `structured_tikz.schema.scene_contract()` / `structured_tikz.validator.validate_scene()`：结构及运行时校验，只用外部 JSON Schema 验证器不能替代后者。
- `structured_tikz.generator.generate_tikz()`：生成可编辑 TikZ；旧 Scene 缺少新增可选样式时保持既有默认行为。
- `scene_structure.recover_scene_structure()`：一次确定性恢复，只处理有证据的结构问题，不补写内容。
- 覆盖及一致性检查是诊断；合法、可编译和节点数量不能证明内容完整。

API、编译和图片反馈的协议测试仍保留。历史输出与原图只在用户本地保存，不是公共安装的必需输入。
