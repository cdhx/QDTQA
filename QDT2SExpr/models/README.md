## model_usage_guide.py
- 如何获得qdt 编码后的数据集，以及获取相关的embedding 模型
- 文件中有一些示例代码，可以作为使用时的参考
### use_serialization()
- 对 QDT 进行序列化编码，一个例子
```json
{"serialization": "[DES] What year [DES] did [INQ] [DES] the basketball team [DES] coached by Brad Stevens [INQ] win the championship"}
```
- 将编码之后的数据送入 BERT 模型，得到 BERT embedding 之后的 qdt