![](https://img.shields.io/badge/license-MIT-blue)
![](https://img.shields.io/badge/python-3.7-green)
![](https://img.shields.io/badge/torch-1.7.1-green)

<h3 align="center">
<p>A PyTorch implementation of neural classififers for Chinese</p>
</h3>

### 1. 已实现模型
- [x] TextCNN
- [x] TextRNN
- [x] Transformer Encoder

---

### 2. 用法

#### 使用参数

```
$ python train.py -h
usage: train.py [-h] [--model_name MODEL_NAME]
                [--output_model_path OUTPUT_MODEL_PATH]
                [--data_path DATA_PATH] [--config_path CONFIG_PATH]
                [--batch_size BATCH_SIZE] [--max_seq_length MAX_SEQ_LENGTH]
                [--lr LR] [--epochs EPOCHS] [--early_stopping EARLY_STOPPING]
                [--display_interval DISPLAY_INTERVAL]
                [--val_interval VAL_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Model name. (default: textcnn)
  --output_model_path OUTPUT_MODEL_PATH
                        Path of the output model. (default: ./output_models/)
  --data_path DATA_PATH
                        Path of the dataset. (default: ./data/)
  --config_path CONFIG_PATH
                        Path of the config file. (default: ./conf/)
  --batch_size BATCH_SIZE
                        Batch size. (default: 64)
  --max_seq_length MAX_SEQ_LENGTH
                        Max sequence length. (default: 100)
  --lr LR               Learning rate. (default: 0.005)
  --epochs EPOCHS       Number of epochs. (default: 10)
  --early_stopping EARLY_STOPPING
                        Early stop. (default: 100)
  --display_interval DISPLAY_INTERVAL
                        Display interval. (default: 1)
  --val_interval VAL_INTERVAL
                        Validation interval. (default: 30)
```

#### 训练
```bash
# TextCNN(default)
$ python train.py --model_name textcnn

# TextRNN
$ python train.py --model_name textrnn

# Transformer Encoder(基于torch.nn实现，简化positional encoding以及输入mask)
$ python train.py --model_name transformer
```

#### 预测
```bash
# 按需修改模型名称和相关模型文件路径
$ python predict.py

箭队史最佳阵容，姚明大梦制霸内线，哈登麦迪完爆勇士水花兄弟                预测类别:sports
平安好医生并不孤单 细数那些从破发开始星辰大海征途的伟大公司               预测类别:entertainment
如果现在由你来接任中国足协主席，你会怎么样做才能提高中国足球整体水平？      预测类别:entertainment
吴广超：5.8伦敦金关注1325争夺继续空，原油择机中空                      预测类别:sports
西仪股份等5只军工股涨停 机构：业绩有望超预期                           预测类别:finance
刘涛：出席活动！网友：我只看到她的一条腿！                             预测类别:entertainment
忘尽心中情，刘德华版《苏乞儿》的主题曲，老歌经典豪气                     预测类别:entertainment
```

#### 数据集
来源于[中文文本分类数据集THUCNews](http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews)，选了体育、娱乐、金融三类数据的新闻标题。

| 数据集划分 | 总样本量 | 娱乐 | 体育 | 金融 |
| :---: | :---: | :---: | :---: | :---: |
| 训练集 | 72000 | 23983 | 23983 | 24034 |
| 验证集 | 9000 | 2965 | 3031 | 3004 |
| 测试集 | 9000 | 3052 | 2986 | 2962 |

| model_name | accuracy |
| --- | --- |
| TextCNN | ~95.38% |
| TextRNN | ~96.19% |
| Transformer | ~97.26% |

---

### 3. 依赖

| package | version |
|--- | --- |
| python | 3.7 |
| torch | 1.7.1 |
| jieba | 0.42.1 |
| tqdm | 4.55.0 |
