# 实现基于深度学习的文本分类器
import logging
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
        self._id2extword = ['[PAD]', '[UNK]']

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
    
    def load_pretrained_embs(self, embfile):
        with open(embfile, encoding='utf-8') as f:
            lines = f.readlines()
            items = lines[0].split()
            word_count, embedding_dim = int(items[0]), int(items[1])

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        for line in lines[1:]:
            values = line.split()
            self._id2extword.append(values[0])
            vector = np.array(values[1:], dtype='float64')
            embeddings[self.unk] += vector
            embeddings[index] = vector
            index += 1

        embeddings[self.unk] = embeddings[self.unk] / word_count
        embeddings = embeddings / np.std(embeddings)

        assign_id = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = assign_id(self._id2extword)

        assert len(set(self._id2extword)) == len(self._id2extword)

        return embeddings
    
    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.unk) for x in xs]
        return self._extword2id.get(xs, self.unk)

    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)
    
    @property
    def word_size(self):
        return len(self._id2word)

    @property
    def extword_size(self):
        return len(self._id2extword)

    @property
    def label_size(self):
        return len(self._id2label)
    
vocab = Vocab(train_data)

w2v_embedding = vocab.load_pretrained_embs('./tmp/word2vec.txt')

################################### 定义DS和DL ###################################
text_max_length = 300

class TextClassifierDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

        
    def _get_embedding(self, id):
        if id < w2v_embedding.shape[0]:
            return w2v_embedding[id]
        else:
            return w2v_embedding[vocab.unk]
        
    def __getitem__(self, index):
        # 每个元素是一篇最大长度为text_max_length的id, 如果不满text_max_length则追加0
        text, label = self.data['text'][index].split()[:text_max_length], self.data['label'][index]
        if len(text) < text_max_length:
            diff_len = text_max_length - len(text)
            text += (['[PAD]'] * diff_len)
        
        text_id = vocab.word2id(text)
        label_id = vocab.label2id(label)

        # print("====>", text_id)
        text_embedding = []
        for id in text_id:
            text_embedding.append(self._get_embedding(id))
        
        return torch.tensor(np.array(text_embedding), dtype=torch.float, device=device), torch.tensor(label_id, dtype=torch.long, device=device)
    
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

################################### 使用LSTM做分类 ###################################
hidden_size = 128
num_classes = vocab.label_size

class TextClassifier(nn.Module):
    def __init__(self) -> None:
        super(TextClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=w2v_embedding.shape[1], hidden_size=hidden_size, dropout=0.2, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])  # 使用最后一个隐藏状态
        return out

################################### 训练和验证 ###################################
num_epochs = 50
learning_rate = 0.001

model = TextClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    model.train()
    for batch_text, batch_label in tqdm(train_ld, total=len(train_ld)):
        outputs = model(batch_text)
        loss = criterion(outputs, batch_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 验证阶段
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    pred_list = []
    label_list = []
    
    with torch.no_grad():
        for texts_batch, labels_batch in valid_ld:
            outputs = model(texts_batch)
            loss = criterion(outputs, labels_batch)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
            
            pred_list += predicted.tolist()
            label_list += labels_batch.tolist()
            
        f1score = f1_score(label_list, pred_list, average='macro')

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss/total:.4f}, Accuracy: {100 * correct / total:.2f}%, F1: {f1score:.4f}')

    with torch.no_grad():
        predicted_list = []
        for texts_batch, labels_batch in test_ld:
            outputs = model(texts_batch)
            _, predicted = torch.max(outputs, 1)
            predicted_list += predicted.tolist()

        out_df = pd.DataFrame({'label': predicted_list})
        out_df.to_csv("./tmp/submit.csv", index=False)