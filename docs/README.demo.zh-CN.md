# 种草鉴别助手 Demo 使用与开发说明

本文对应 2026-10-12 的本地 Demo 界面，功能基线为提交 `ccd822a`。
项目仓库：https://github.com/jiaweiz5/projectdounai

## 1. 这个 Demo 做什么

这是面向小红书笔记的辅助分析原型。界面分为两个独立入口：

| 入口 | 用户想解决的问题 | 输入 | 展示结果 |
| --- | --- | --- | --- |
| AI 文案鉴别 | 这段文字有没有 AI 参与写作的迹象？ | 笔记正文 | AI 写作特征判断和模型估计分数 |
| 种草推广核验 | 这篇种草有哪些值得复核的推广线索？ | 正文必填；配图、评论和可信原图选填 | 推广表达、评论协同、图文一致性、原图差异四项独立结果 |

真人写作也可能包含推广，AI 写作也不一定是广告，因此两种需求分别展示。
“种草推广核验”不会把四项结果相加生成一个综合推广概率。
图片不匹配、评论相似或图片存在改动，都不能单独证明商业合作。

## 2. 如何使用

### AI 文案鉴别

1. 选择「AI 文案鉴别」。
2. 粘贴正文，或导入 UTF-8 编码的 `.txt` 文件；输入 20–10,000 字，尽量保留完整上下文。
3. 点击「开始鉴别」，也可以使用 `Ctrl / ⌘ + Enter`。
4. 查看判断与分数，按需下载结果。

「试用示例」用于体验流程，不是真人来源已核实的样本。
模型分数是对整篇文字的估计，不是“有多少百分比的文字由 AI 写成”。

### 种草推广核验

1. 选择「种草推广核验」，填写笔记正文。
2. 有配图时展开「笔记配图」并上传，系统会比较图片与正文，同时识别图中文字。
3. 有可信原图时，在对应配图下展开「添加可信原图，检查图片改动」，选择同一画面的原图。
4. 有评论时展开「评论区」，上传截图或手动添加评论。
5. 检查评论列表后点击「开始核验」，查看右侧逐项报告。

没有配图或评论也可以先核验正文。缺少材料会显示“未提供”；服务失败会显示“未完成”，都不表示零风险。
当前页面中的材料保存在页面状态中，切换两个入口会保留，刷新页面会清空。需要留存时请下载结果。

### 评论截图 OCR 的完整流程

```text
选择评论截图 → 识别并整理评论 → 检查、修改、删除或补充评论
→ 勾选“我已检查评论分段和文字” → 开始核验 → 查看实际分析评论数
```

- 尽量上传清晰、裁剪到评论区域的原始截图。
- OCR 先识别文字，再结合日期、回复标记和行位置整理评论段落。
- 一条评论的多行文字会尝试合并；没有日期的末尾评论使用版面信息辅助整理。
- 真正重复出现的评论会保留，因为重复本身是评论协同分析的输入。
- 识别可能漏字、误分段或混入用户名。用户可以直接修改每条评论，也可以展开「查看识别原文，补充遗漏评论」。
- 多张截图有重叠时，请人工删除重复导入的同一条评论；不要删除原评论区里实际存在的重复评论。
- 至少 5 条可用评论才能进行协同分析，最多 30 条。少于 5 条时，其他项目仍可运行。

### 上传限制

| 材料 | 当前界面限制 |
| --- | --- |
| 笔记正文 | 最多 10,000 字 |
| 笔记配图 | JPG / PNG / WebP；最多 2 张，合计不超过 2 MB |
| 评论截图 | JPG / PNG / WebP；单张不超过 12 MB |
| 可信原图 | JPG / PNG / WebP；单张不超过 12 MB |
| 评论 | 最多 30 条；每条编辑框最多 5,000 字符 |

这些是当前前端限制，后端还会校验文件和图片尺寸。过大的图片即使通过前端检查，也可能被后端拒绝。

## 3. 与四层后端的对应关系

| 后端模块 | 界面位置 | 用途与边界 |
| --- | --- | --- |
| Layer 1 | AI 文案鉴别 | 判断写作特征，不能确定作者实际使用了哪款 AI |
| Layer 2 | 种草推广核验 → 推广表达 | 分析正文的推广倾向，不能直接证明未披露的商业合作 |
| Layer 3 | 种草推广核验 → 评论协同 | 分析整组评论的重复、相似话术，不能据此确认账号属于水军 |
| Layer 4A | 种草推广核验 → 图文一致性 | 比较图片与正文，并展示图片 OCR；匹配分数不是概率 |
| Layer 4B | 种草推广核验 → 原图差异 | 比较待核验图与可信原图；裁剪、滤镜、压缩也可能产生差异 |

当前图片功能不是 AI 生成图片鉴定，也不能仅凭商品配图证实功效。
界面改版没有重新训练模型或调整后端阈值。

## 4. 技术结构与接口

前端使用 Next.js、React、TypeScript 和 next-intl；后端使用 Python FastAPI。
浏览器把材料交给 Next.js 服务端代理，再转发给 FastAPI。OCR 与检测模型在后端执行。

| 前端接口 | 后端接口 | 请求内容 |
| --- | --- | --- |
| `POST /api/analyze` | `POST /analyze` | JSON：`text`、`comments`、`images`、`include_report` |
| `POST /api/analyze-screenshot` | `POST /analyze-screenshot` | multipart：`file` |
| `POST /api/analyze-layer4-reference` | `POST /analyze-layer4-reference` | multipart：`file` 和 `reference_file` |

主分析中的 `images` 是图片 data URL 数组，`comments` 是检查后的评论字符串数组。
当前界面设置 `include_report: false`，展示检测器结构化结果，不请求 Qwen 生成解释报告。
AI 文案鉴别也使用现有 `/analyze` 接口，但只展示其中的 authorship 结果。
OCR 接口的初步分析不会代替用户检查后的最终评论分析。

主要文件：

```text
frontend/src/app/page.tsx                     两个入口、AI 文案鉴别
frontend/src/app/verifier.css                 共用卡片、栏宽、输入框、按钮样式
frontend/src/components/NoteCheckWorkspace.tsx 推广核验、材料整理和报告
frontend/src/components/NoteCheckWorkspace.module.css
frontend/src/app/api/                         后端代理
backend/main.py                              FastAPI 接口
backend/screenshot_ocr.py                     截图 OCR 与评论分段
backend/tests/test_screenshot_ocr.py           分段回归测试
```

两页共用 `detection-grid`、`editor-card`、`main-editor`、`result-panel` 等样式。
后续调整栏宽、正文框、圆角、焦点边框或手机断点时，应优先修改共用规则，避免两页再次不一致。
配图与评论是输入卡片内部的折叠区，可信原图属于具体配图。

## 5. 本地运行

### 当前这台 Windows 电脑

运行副本位于 `E:\美妆数据\demo`。已有依赖和模型环境时，在 PowerShell 中执行：

```powershell
cd E:\美妆数据\demo
.\start-local.ps1
```

显示 `READY` 后访问 http://127.0.0.1:3000 。启动日志位于 `.local-logs`。
该启动脚本是当前电脑的本地便利脚本，不能假设 GitHub 克隆中也带有它或相同的运行环境。

### 队友在另一台电脑运行

先安装 Node.js/npm 和 Python，并准备与模型训练环境兼容的依赖版本。
克隆后需要完整的 `backend/models` 及相关配置，而不只是前端源码。
Layer 4 使用 `backend/data/layer4/layer4_config.json` 和原图比对模型；部分预训练模型首次使用时还需要联网下载。
缺少模型时，应向模型负责人取得对应版本，不要为了启动页面临时重训或改阈值。

```powershell
git clone https://github.com/jiaweiz5/projectdounai.git
cd projectdounai
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
# Layer 4 引用了 cv2；当前 requirements.txt 未显式列出该依赖
.\.venv\Scripts\python.exe -m pip install opencv-python-headless
cd backend
..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

另开一个 PowerShell 窗口，从仓库根目录启动前端：

```powershell
cd frontend
npm ci
npm run dev -- --hostname 127.0.0.1 --port 3000
```

macOS/Linux 使用 `.venv/bin/python` 替代 `.venv\Scripts\python.exe`，其余流程相同。
以上是项目安装流程说明，本次没有在全新电脑上重新验证完整安装。

前端默认连接 `http://127.0.0.1:8000`。需要修改时，在 `frontend/.env.local` 设置后重启前端：

```ini
BACKEND_URL=http://127.0.0.1:8000
```

- 前端：http://127.0.0.1:3000
- 后端健康检查：http://127.0.0.1:8000/health
- 后端接口文档：http://127.0.0.1:8000/docs

远程部署不能直接照搬本地配置：截图与原图代理对非本机后端要求 `BACKEND_SHARED_TOKEN`，而当前主分析代理没有转发该令牌。接入需要鉴权的远端后端时，需先统一接口鉴权。密钥仅放在服务端环境变量中，不提交到仓库。

## 6. 联调与常见问题

| 现象 | 优先检查 |
| --- | --- |
| 网页打不开 | 前端是否启动，3000 端口是否被其他程序占用 |
| 网页打开但检测失败 | 后端健康检查、8000 端口和 `BACKEND_URL` |
| OCR 提示失败 | 截图大小和格式、代理返回状态、后端 OCR 日志及模型是否就绪 |
| 评论只有一行或混入用户名 | 检查截图清晰度、日期是否识别成功；在可编辑列表中修正 |
| 评论数为 0 | 是否只上传了截图、尚未识别并加入列表；最终请求是否包含评论 |
| 图文或原图结果未完成 | Layer 4 模型、配置、OpenCV 依赖及首次下载状态 |
| 代码已改，后端行为没变 | 当前本地启动脚本没有开启后端自动重载，需要重启后端 |

建议交付前依次验证：AI 文案检测；仅正文推广核验；截图识别后编辑并提交至少 5 条评论；正文加配图；配图加可信原图；手机宽度与错误提示。

```powershell
# 在 backend 目录执行
..\.venv\Scripts\python.exe -m unittest discover -s tests -p test_screenshot_ocr.py

# 在 frontend 目录执行
npm run build
```

本地已完成构建、OCR 分段回归测试和浏览器联合流程验证。测试截图的 6 条评论均进入最终分析，多行正文与真实重复评论得到保留；图片 OCR、图文一致性和原图比较接口均返回了结果。
这些验证说明对应流程能运行，不代表对所有小红书内容都识别准确，也不构成新的模型准确率评估。

## 7. 协作说明

当前电脑的 Git 协作目录为 `E:\美妆数据\projectdounai-collab`，与运行目录 `E:\美妆数据\demo` 分开。
运行目录里的修改不会自动进入 GitHub。交付时应确认修改已进入 Git 仓库，并同步到实际运行副本验证。
仓库更新前先检查本地改动并拉取远端，避免覆盖队友工作。不要提交虚拟环境、依赖目录、日志、密钥或临时测试截图。

本文补充现有项目 README，重点说明 Demo 的实际操作与前后端连接；模型训练、数据来源和评估方法以对应模型文档为准。
