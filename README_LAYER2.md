Paste this into the Layer 2 section of your README:

### Layer 2 — Covert Advertisement Detection

The detector uses character-level TF-IDF features with class-balanced logistic regression and a frozen decision threshold of 0.55. On the 328-post locked test set, it achieved **87.2% accuracy, 77.1% ad precision, 54.4% ad recall, and a 63.8% F1 score**. The confusion matrix was **TN 249, FP 11, FN 31, TP 37**. End-to-end API checks passed **6/6**, including the correct detection of a held-out CHASM ad. The model missed 31 of the 68 ads in the locked test, so `likely_non_ad` is only a model prediction; it does not prove that a post contains no advertising.

### 第二层：隐性广告检测

该检测器使用字符级 TF-IDF 特征和类别权重平衡的逻辑回归模型，判定阈值固定为 0.55。在包含 328 条帖子的锁定测试集上，模型的**准确率为 87.2%、广告精确率为 77.1%、广告召回率为 54.4%、F1 分数为 63.8%**。混淆矩阵结果为 **TN 249、FP 11、FN 31、TP 37**。端到端 API 测试 **6/6 通过**，其中包括一条被正确识别的 CHASM 留出广告样本。模型在锁定测试集中漏掉了 68 条广告中的 31 条，因此，`likely_non_ad` 只是模型预测结果，不能证明帖子中完全没有广告。