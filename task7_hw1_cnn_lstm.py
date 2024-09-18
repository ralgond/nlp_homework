# 实现基于深度学习、Transformer的文本分类器
import logging
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

# set cuda
gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
else:
    device = torch.device("cpu")
logging.info("Use cuda: %s, gpu id: %d.", use_cuda, gpu)

#set seed
seed = 666
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available() >= 0:
    torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

################################### split data to 10 fold ###################################
fold_num = 10
data_file = './data/train_set.csv'

def all_data_2_fold(fold_num, num=200000):
    assert num % fold_num == 0

    fold_data = []

    df = pd.read_csv(data_file, sep='\t', encoding='UTF-8', nrows=num)
    
    np.random.shuffle(df.values)

    texts = df['text'].tolist()
    labels = df['label'].tolist()

    step = num // fold_num
    for start_pos in range(0, num, step):
        # print(start_pos, start_pos+step)
        fold_data.append({'text':texts[start_pos:start_pos+step], 'label':labels[start_pos:start_pos+step]})

    return fold_data


fold_data = all_data_2_fold(10)

################################### build train, dev, test data ###################################
fold_id = 9

# dev
dev_data = fold_data[fold_id]

# train
train_texts = []
train_labels = []
for i in range(0, fold_id):
    data = fold_data[i]
    train_texts.extend(data['text'])
    train_labels.extend(data['label'])

train_data = {'label': train_labels, 'text': train_texts}

# test
test_data_file = './data/test_a.csv'
f = pd.read_csv(test_data_file, sep='\t', encoding='UTF-8')
texts = f['text'].tolist()
test_data = {'label': [0] * len(texts), 'text': texts}


################################### build vocab ###################################
class Vocab():
    def __init__(self, train_data) -> None:
        self.min_count = 5
        self.pad = 0
        self.unk = 1
        self._id2word = ['[PAD]', '[UNK]']

        self._id2label = []
        self.target_names = []

        self.build_vocab(train_data)

        assign_id = lambda x: dict(zip(x, range(len(x))))
        self._word2id = assign_id(self._id2word)
        self._label2id = assign_id(self._id2label)

        logging.info("Build vocab: words %d, labels %d." % (self.word_size, self.label_size))

    def build_vocab(self, data):
        self.word_counter = Counter()

        for text in data['text']:
            words = text.split()
            for word in words:
                self.word_counter[word] += 1
        
        for word, count in self.word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word) # 按照word的出现次数的逆序

        label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政', 5: '社会', 6: '教育', 7: '财经',
                      8: '家居', 9: '游戏', 10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}
        
        self.label_counter = Counter(data['label'])

        for label in range(len(self.label_counter)):
            count = self.label_counter[label]
            self._id2label.append(label)
            self.target_names.append(label2name[label])
    
    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)

    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)
    
    @property
    def word_size(self):
        return len(self._id2word)

    @property
    def label_size(self):
        return len(self._id2label)
    
vocab = Vocab(train_data)


################################### 定义DS和DL ###################################
text_max_length = 400

class TextClassifierDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        
    def __getitem__(self, index):
        # 每个元素是一篇最大长度为text_max_length的id, 如果不满text_max_length则追加0
        text, label = self.data['text'][index].split()[:text_max_length], self.data['label'][index]
        if len(text) < text_max_length:
            diff_len = text_max_length - len(text)
            text += (['[PAD]'] * diff_len)
        
        text_id = vocab.word2id(text)
        label_id = vocab.label2id(label)
        
        return torch.tensor(np.array(text_id), dtype=torch.long, device=device), torch.tensor(label_id, dtype=torch.long, device=device)
    
    def __len__(self):
        return len(self.data['text'])
    
train_ds = TextClassifierDataset(train_data)

train_ld = DataLoader(train_ds, batch_size=64, shuffle=True)

valid_ds = TextClassifierDataset(dev_data)

valid_ld = DataLoader(valid_ds, batch_size=64, shuffle=False)

test_ds = TextClassifierDataset(test_data)

test_ld = DataLoader(test_ds, batch_size=64, shuffle=False)

for batch_text, batch_label in valid_ld:
    print (batch_text.shape)
    print (batch_label.shape)
    break


################################### Transformer 分类模型 ###################################
class CNN_LSTM_Classifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, dropout=0.2, num_filters=100, kernel_size=3, hidden_size=128):
        super(CNN_LSTM_Classifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.conv = nn.Conv2d(1, num_filters, (kernel_size, embed_size))
        self.lstm = nn.LSTM(num_filters, hidden_size, dropout=0.2, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        # Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # [batch_size, 1, seq_len, embed_size]
        x = torch.relu(self.conv(x)).squeeze(3)  # [batch_size, num_filters, new_seq_len]
        
        # 转换维度以匹配 LSTM 输入，放入 LSTM 处理时序特征
        x = x.permute(0, 2, 1)  # [batch_size, new_seq_len, num_filters]

        # LSTM 层：捕捉序列中的长距离依赖关系
        x, (hn, cn) = self.lstm(x)  # [batch_size, new_seq_len, hidden_size]

        # 取最后一个时间步的隐藏状态
        x = hn[-1]  # [batch_size, hidden_size]

        # Dropout
        x = self.dropout(x)

        # 全连接层：进行分类
        output = self.fc(x)  # [batch_size, num_classes]
        return output
    
################################### 4. 模型训练 ###################################
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data, labels in tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        val_loss = 0
        correct = 0
        total = 0
        
        pred_list = []
        label_list = []
    
        model.eval()
        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pred_list += predicted.tolist()
                label_list += labels.tolist()
                
            f1score = f1_score(label_list, pred_list, average='macro')

        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, Val Accuracy: {100 * correct / total}%, F1: {f1score:.4f}')
        
        with torch.no_grad():
            predicted_list = []
            for texts_batch, labels_batch in test_ld:
                outputs = model(texts_batch)
                _, predicted = torch.max(outputs, 1)
                predicted_list += predicted.tolist()

            out_df = pd.DataFrame({'label': predicted_list})
            out_df.to_csv("./tmp/submit.csv", index=False)



################################### 7. 创建模型并进行训练 ###################################
embed_size = 200  # 嵌入层维度
num_heads = 4  # 多头注意力的头数
num_encoder_layers = 3  # Transformer 编码器层数
max_len = 512  # 最大序列长度

model = CNN_LSTM_Classifier(vocab.word_size, embed_size, vocab.label_size).to(device)


################################### 训练模型 ###################################
train_model(model, train_ld, valid_ld, num_epochs=20)
