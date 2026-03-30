# 作业4：电影评论情感分类结果汇总（自动生成）

| model | accuracy | precision | recall | f1 | train_seconds | infer_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression(TF-IDF) | 0.9008 | 0.8989 | 0.9030 | 0.9010 | 0.5707 | 0.0049 |
| LinearSVM(TF-IDF) | 0.8912 | 0.8920 | 0.8902 | 0.8911 | 0.5898 | 0.0085 |
| RNN-LSTM(Embedding, device=cpu) | 0.5128 | 0.5549 | 0.1290 | 0.2093 | 90.0339 | 17.4576 |
