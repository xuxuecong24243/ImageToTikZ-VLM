# 几何候选到人工审校场景

本流程不识别文字、不推断张量、箭头或拓扑，不调用 API。保持现有生成器和 Schema 不变。

## 创建初稿和审校图

在项目根目录运行，输出目录使用新的名称：

```powershell
.\.venv\Scripts\python.exe -B scene_builder.py --geometry "outputs/geometry_20261112T132317Z_323267fa/geometry.json" --output "outputs/scene_review_new/scene_draft.json" --reference "examples/gpt_embedding_scene.json"
.\.venv\Scripts\python.exe -B scene_review.py --scene "outputs/scene_review_new/scene_draft.json" --image "inputs/test.png"
```

`--reference` 仅计算报告中的 bbox 匹配，不向初稿添加节点类型、标签、网格或连接。四个张量的人工参考框同时比较正面和整体投影范围，防止把整体检测框直接当作正面。

自动转换保留像素 bbox 及候选 ID，所有节点均为 `rectangle`、`unverified`。无填充的深色边框仅是审校默认样式。像素坐标优先；只有归一化数据时按宽高换算并注明。两套坐标矛盾、非法对象或不支持的源版本会报错，不静默丢弃。

每个节点的 `metadata.source_candidate` 保存原始记录。顶层 `metadata.review_candidates` 保留可选线段、被排除的文字笔画、未识别文字框及几何分组；其余源字段保存在 `source_geometry_metadata`。检测分数不是识别正确率。

## JSON 人工编辑

1. 按图检查各节点，修改 bbox；必要时改为 matrix/tensor3d，补齐 grid 和 depth。张量 bbox 必须是正面，不是包含顶面和侧面的检测框。
2. 在 `edges` 中人工补充连接。可复制候选的 `points`，指定 `arrow`；也可使用节点 ID 和锚点。没有自动箭头方向或 from/to 猜测。
3. 在 `labels` 中补充中文和数学文本。文字框的 `recognized_text:null` 不代表已完成 OCR。
4. 每个实际绘图对象的 `metadata.review_status` 设为 `verified` 或 `rejected`；确认整体后才把顶层状态改为 `verified`。辅助候选不是绘图对象，不必全部转换为边。

Python 辅助接口 `scene_builder.edge_from_candidate(scene, candidate_id, edge_id)` 只返回人工选中的、无箭头的 `unverified` 边，调用者再加入 edges 并审校。不会修改输入，也不会复用已排除的文字笔画。

审校预览为橙色待审节点、绿色已确认节点、红色拒绝节点，只绘制已确认的 edges。不把“Schema 合法”当作“已人工审核”。

## 导出经过声明的审核版本

人工编辑完成后，使用新文件名；以下仅为后续操作示例：

```powershell
.\.venv\Scripts\python.exe -B scene_review.py --scene "outputs/scene_review_new/scene_draft.json" --export-verified "outputs/scene_review_new/scene_verified.json" --confirm-reviewed --reviewer "你的名字"
.\.venv\Scripts\python.exe -B -m structured_tikz.generator --input "outputs/scene_review_new/scene_verified.json" --image "inputs/test.png"
```

导出需要明确确认、审核人、顶层 verified 及每个绘图对象的处理状态。导出会移除 rejected 对象，并再次调用现有 Validator 检查引用、尺寸及数学命令，不能留下指向已拒绝节点的边。

这是本地人工声明，不是身份认证或审核证据验证。旧生成器只校验 Schema，因此应通过此导出步骤后再将场景交给生成器。本轮实验仅生成未审核初稿和预览，不生成冒充已审核的 scene_verified.json，也不编译不完整的场景。
