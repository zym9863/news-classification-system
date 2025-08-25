# Backend 使用说明补充

## 数据列名要求

分类训练默认从 `data.xlsx`（或传入路径）加载数据。文本列与标签列支持自动解析以下常见别名（大小写不敏感）：

- 文本列别名：`标题`、`题目`、`新闻标题`、`内容`、`文本`、`文章`、`摘要`、`title`、`headline`、`text`、`content`
- 标签列别名：`类别`、`分类`、`标签`、`类目`、`类型`、`种类`、`label`、`category`、`type`、`class`

若数据表列名不在上述范围，建议：
1. 将列名改为常见别名之一；或
2. 在初始化分类器时显式指定列名：

```python
from models.classifier import NewsClassifier

clf = NewsClassifier(
	data_path="my_data.xlsx",
	text_column="我的标题列",
	label_column="我的标签列",
)
clf.train_model()
```

此外，`data_path` 支持 `.xlsx/.xls/.csv`（自动按扩展名选择读取）。

