## 基于 torchtext 处理自然语言问题输入
### serializationDataset.py: 将 QDT 进行序列化
- 对于 conjunction，在 description **开始**加上 [DES] token
- 对于 composition 的内层问题，在内层问题**两侧**加上 [IQS] token
- token 不能和 BERT 预设的冲突；Hugging Face addSpecialToken
#### qdt_serialization 
- 应该是一个递归函数
- 是一个 item, 且没有 inner_questions 属性: [DES] + decription
- 是一个 item, 但是有 inner question： [INQ] recur(children) [INQ]
- 有多个 item： recur(child1) + recur(child2) ...
- 测试: 完成 test 上 qdt 的读取，全部用这个函数转换一下，看一下转换结果是不是预想的
### 对序列化后的 QDT 进行编码，得到数据集
- 友好的接口，参数别写死
- 添加了两个特殊 token: [DES] 和 [INQ]
- 抽查了一下 vocab, 确认这些特殊 token 被正确编码
  - 最好能够把编码的内容解码回来，看看和原句是否一致
- json encoding 问题
  - 文件需要是 jsonl 格式，也就是每个 dict 之间用 '\n' 分隔
  - field 长这样 fields = {'serialization': ('serialization', self.SERIALIZATION)}
  - 不支持 text field (对于 text 会自动编码)，所以我们不能用数据集中的 "ID" 作为表示，得新弄一个数字的 "idx"
- 看一下数据集里头最长的句子有多少 token (30 27 30), 考虑到 BERT 头尾还要各加上一个 token, 所以推荐 fix_length=32