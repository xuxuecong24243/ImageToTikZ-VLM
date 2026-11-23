# 确定性场景绘图

仅将已给定的图形结构转成 TikZ，不识别图片，不导入 VLM 客户端。

在项目根目录执行（无需 API Key）：

```powershell
.\.venv\Scripts\python.exe -B -m structured_tikz.generator --input "examples/gpt_embedding_scene.json" --image "inputs/test.png"
```

复用项目原有中文模板、XeLaTeX 编译器及 PDF/PNG 对比模块。`--image` 仅用于编译后的左右对照，不进入代码生成器。可使用 `--xelatex` 指定编译器，`--timeout` 控制秒数。每次运行写入新的 `outputs/structured_*` 目录，保留场景副本、TikZ、完整 TEX、PDF、PNG、日志及 run.json；失败不自动修复或重试。

## 场景约定

- `canvas`: `width`、`height` 为原图像素，`coordinate_system` 必须为 `top-left`；可选 `scale` 为 pt/像素，默认 0.4。所有几何、字体大小、线宽和圆角半径均用像素表示。
- `styles`: 名称到样式字典的映射。对象的 `style` 可直接给字典或引用名称。颜色仅允许 `schema.py` 中的固定颜色；没有任意 TikZ 选项透传。
- `nodes`: `id,type,bbox=[x,y,w,h]`。支持 `rectangle`、`rounded_rectangle`、`matrix`、`tensor3d`、`text`。矩阵及张量的 `grid={rows,cols}` 表示绘制的示意网格，不等同于数学维度。
- `tensor3d`: bbox 为正面；`depth={dx,dy}` 为背面相对正面的像素偏移，目前限定 dx>0、dy<0，即向右上。正面宽高、深度及网格可分别修改。
- `edges`: 通过 `from,to,from_anchor,to_anchor` 引用节点，或使用完整的 `points` 折线路径，两种方式互斥。锚点支持 north/south/east/west。`routing=orthogonal` 时 `first_axis=x|y` 明确第一个方向，默认 x；不会自动绕障。`arrow=none|start|end|both`，`dashed` 为布尔值。
- `labels`: `id,text,x,y`；可选 `anchor,font_size,align,mode,style`。默认字号22像素、居中、普通文字；`mode=math` 的 LaTeX 不带 `$`，反斜杠遵循 JSON 转义。
- 节点的可选 `text` 也生成独立文字节点；`text` 类型节点按 bbox 宽度换行，独立 labels 不自动换行。样式的 `bold` 控制加粗。
- `metadata` 仅保留说明，不参与 LaTeX 生成。示例语义和坐标来自人工看图标注，不是 OpenCV 识别结果。

统一映射为 `x'=scale*x`，`y'=scale*(H-y)`，用显式 pt 单位。张量顶点先在同一个像素坐标系构造，再统一转换。绘制顺序为 nodes → edges → 节点文字 → labels，各数组保持输入顺序。图形节点名为 `n-ID`，独立标签为 `l-ID`，节点自带文字为 `t-ID`。

普通文字逐字符转义；数学内容只允许白名单命令、平衡花括号和有限长度，随后再次经过现有 TikZ 安全检查。校验器拒绝未知字段、非法引用及非有限数值。这不构成 TeX 安全沙箱，也不是完整的数学语法检查器；公开处理不可信输入仍需隔离、权限限制和资源配额。

## 实验边界

人工样例覆盖主要矩阵、四个张量、中文标签、数学维度和显式连线。坐标为人工近似值，字号与字体度量可能造成与原图的差异。确定性生成避免模型改写坐标和连接关系，但不能修复错误的输入标注，也不自动检查全部文字重叠。尺寸花括号等装饰未纳入本轮图元范围。
