# 千问评论截图识别

评论截图由阿里云百炼 `qwen3.5-flash` 读取，不再调用本地 OCR。
正文分类、推广分类和图片图文一致性模型保持原有实现；图片模块自己的文字提取不属于本次替换范围。

## 队友本地接入

在后端的 `backend/.env` 中配置（不要提交该文件）：

```dotenv
DASHSCOPE_API_KEY=填写你自己的百炼密钥
QWEN_VISION_MODEL=qwen3.5-flash
QWEN_VISION_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

接口地址必须与密钥所属地域一致。也可用 `QWEN_VISION_API_KEY` 单独配置视觉接口密钥，其优先级高于 `DASHSCOPE_API_KEY` 和旧的 `QWEN_API_KEY`。
安装原有后端依赖后重启后端即可，不需要额外下载识别模型。不要将密钥放入前端或 `NEXT_PUBLIC_*` 变量。
每次识别会把所选截图发送至百炼并使用该密钥账户的额度；团队共用已部署后端时，只需在服务器配置一次。

## 流程和接口

上传截图 → 千问整理完整评论 → 用户编辑、确认 → 点击核验 → 正文及评论模型分别分析。
重复评论保留。无法辨认的文字标为 `[无法辨认]`，识别失败不会退回本地 OCR，也不会生成零风险结论。

`POST /analyze-screenshot` 接收 multipart `file`。为兼容已有前端，返回字段仍名为 `ocr`，内有 `provider`、`model`、`comments`、`lines` 和数量；此字段名不表示运行本地 OCR。
该接口不再直接返回 Layer 3 分数，评论确认后通过 `/analyze` 提交。最多加入30条评论，至少5条才进行协同分析。
模型转录仍可能遗漏或误识别，请核对后再分析。
