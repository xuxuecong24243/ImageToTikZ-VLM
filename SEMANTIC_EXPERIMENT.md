# 单独测试视觉语义理解

本入口只读取原图与 OpenCV 场景候选，要求模型返回语义 JSON。不会生成 TikZ、编译或自动审核；人工参考文件没有 CLI 输入选项，也不会被读取用于 Prompt。

## 离线准备（默认）

```powershell
.\.venv\Scripts\python.exe -B semantic_analyzer.py --image "inputs/test.png" --scene "outputs/scene_draft_20261112T142032Z/scene_draft.json" --thinking-mode disabled --max-tokens 32768
```

不需要 API Key。每次新建 `outputs/semantic_*`，保存：

- `candidate_annotation.png`：与原图同尺寸，无坐标偏移，Python 绘制候选框及 ID。
- `semantic_context.json`：实际输入的候选 ID/bbox、最多12条线段提示、文字区域位置。没有识别文本。
- `semantic_prompt.txt`：独立的语义识别提示及严格响应契约。
- `run.json`：配置检查、授权和运行状态。

`--check` 只显示配置状态，不生成文件、不发请求。配置通过现有 `API_BASE_URL`、`API_KEY`、`MODEL_NAME`，以及原有 Thinking/输出预算环境变量读取；没有默认模型，不自动加载 .env。

## 后续真实实验（本轮不执行）

只有在明确允许付费调用后才执行：

```powershell
.\.venv\Scripts\python.exe -B semantic_analyzer.py --image "inputs/test.png" --scene "outputs/scene_draft_20261112T142032Z/scene_draft.json" --allow-api --thinking-mode disabled --max-tokens 32768
```

CLI 和客户端均检查授权。每次最多一个请求，不重试、不修复失败 JSON。复用现有 VLMClient 的配置、HTTP、Thinking、预算、诊断、截断拒绝和脱敏逻辑。原图与标注图按两张独立图片发送，原始图片字节不变。没有依赖服务商专有 JSON 模式，是否稳定输出严格 JSON 仍需真实实验验证。

## 响应契约及合并

`semantic_schema.CONTRACT` 是完整契约说明。顶层必须包含 `objects`、`relations`、`texts`、`unmatched_objects`、`unmatched_texts`、`warnings` 六个数组。

- 对象只能引用已有候选 ID；类型为 rectangle/rounded_rectangle/matrix/tensor3d/text/unknown。禁止返回对象 bbox 修改、grid 或 depth。
- 关系区分 direct_line/arrow/proximity/unknown；方向独立记录为 forward/reverse/bidirectional/none/unknown。不根据邻近推断箭头。
- 文字独立保留 raw_text、text、mode、bbox、candidate_id、text_region_id。位置不明确时 bbox 为 null；数学 text 不带 `$`，须通过现有数学命令白名单。复杂公式若无法可靠转换，应保留原始转写并报告不确定性。
- 所有条目 review_required 必须为 true；confidence 为模型自报 high/medium/low/unknown，不是经过校准的准确率。
- 拒绝 Markdown、重复 JSON 键、重复对象、非法 ID、未知字段、非有限数值和不安全数学命令。

合并只写新的 `scene_semantic_draft.json`，不改输入场景。原节点 ID、bbox、已有 edges 和 labels 保留。已 verified/rejected 的对象不改写；整份已审核场景拒绝自动合并。

如果类型切换无需编造绘图参数且符合现有 Schema，可应用为待审类型；matrix/tensor3d/text 缺少必要参数时只记录建议及延后原因。尤其不把包含三维投影的框直接转换为张量正面。

有可靠文字位置才创建待审标签。引用已知文字区域时使用原始测量位置；否则将模型 bbox 标为估计值。字号22是明确的审校默认值，不是识别结果。所有关系、未匹配对象及无位置文字保留在 metadata.semantic_review，不强行转为最终 edges/nodes。

成功时额外保存脱敏的 `semantic_response.txt`、`semantic_result.json`、`scene_semantic_draft.json`。失败保留诊断；只有显式 `--save-api-diagnostics` 才留存可见的截断文本，不保存隐藏推理内容或密钥。

本轮测试使用明确标识的合成 Mock 内容，不能证明中文、公式或连接识别正确。人工参考 JSON 仅应在收到真实结果后做独立本地评估；本轮没有实现自动评分，也没有把参考答案注入请求。人工审核仍沿用 `SCENE_REVIEW.md`，不会自动创建 scene_verified.json。
