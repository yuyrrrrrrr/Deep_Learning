import jieba
import torch
from sympy import print_tree
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm


def load_data():
    unique_words = []
    all_words = []

    for line in open("jaychou_lyrics.txt", mode="r", encoding="utf-8"):
        words = jieba.lcut(line)
        all_words.append(words)
        for word in words:
            if word not in unique_words:
                unique_words.append(word)

    word_count = len(unique_words)

    word_to_index = {word: idx for idx, word in enumerate(unique_words)}

    corpus_idx = []

    for words in all_words:
        temp = []
        for word in words:
            temp.append(word_to_index[word])
        temp.append(word_to_index[" "])

        corpus_idx.extend(temp)

    return unique_words, word_to_index, word_count, corpus_idx


def tst_load_data():
    unique_words, word_to_index, word_count, corpus_idx = load_data()
    print("unique_words--->", unique_words)
    print("word_to_index--->", word_to_index)
    print("word_count--->", word_count)
    print("corpus_idx--->", corpus_idx)


class LyricDataset(Dataset):
    def __init__(self, corpus_idx, num_chars):
        self.corpus_idx = corpus_idx
        self.num_chars = num_chars
        self.word_count = len(self.corpus_idx)
        self.number = self.word_count // self.num_chars

    def __len__(self):
        return self.number

    def __getitem__(self, item):
        item = min(max(item, 0), self.word_count - self.num_chars - 1)
        x = self.corpus_idx[item: item + self.num_chars]
        y = self.corpus_idx[item + 1: item + 1 + self.num_chars]
        return torch.tensor(x), torch.tensor(y)


def tst_Lyricdataset():
    unique_words, word_to_index, word_count, corpus_idx = load_data()
    print("corpus_idx--->", corpus_idx)
    dataset = LyricDataset(corpus_idx=corpus_idx, num_chars=5)
    print("dataset--->", dataset[0])


class LyricModel(nn.Module):
    def __init__(self, word_count):
        super().__init__()
        self.embedded = nn.Embedding(num_embeddings=word_count, embedding_dim=128)
        self.rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)
        self.out = nn.Linear(in_features=256, out_features=word_count)

    def forward(self, input, hidden):
        embedded = self.embedded(input)
        output, hidden = self.rnn(embedded.transpose(0, 1), hidden)
        output = self.out(output.reshape(-1, output.shape[-1]))

        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, 256)


def train_model():
    unique_words, word_to_index, word_count, corpus_idx = load_data()
    dataset = LyricDataset(corpus_idx=corpus_idx, num_chars=32)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, drop_last=True)
    model = LyricModel(word_count=word_count)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 2
    total_loss = 0
    total_num = 0
    start_time = time.time()
    for epoch in range(epochs):
        for x, y in tqdm(dataloader):
            hidden = model.init_hidden()
            output, hidden = model(x, hidden)

            loss = criterion(output, y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_num += len(y)
        print("epoch--->", epoch + 1, "avg_loss:", total_loss / total_num, "time:", time.time() - start_time)
    torch.save(model.state_dict(), "./lyric_model.pth")


def generate_lyric(start_word, sentence_length):
    unique_words, word_to_index, word_count, corpus_idx = load_data()

    model = LyricModel(word_count=word_count)

    model.load_state_dict(torch.load("./lyric_model.pth"))
    model.eval()
    hidden = model.init_hidden()

    word_index = word_to_index[start_word]
    generate_words = [word_index]

    for _ in range(sentence_length):
        output, hidden = model(torch.tensor([[word_index]]), hidden)

        word_index = torch.argmax(output, dim=-1)
        generate_words.append(word_index.item())

    print("generate_words--->", generate_words)
    print("len(generate_words)--->", len(generate_words))
    for idx in generate_words:
        print(unique_words[idx], end='')


if __name__ == '__main__':
    # train_model()
    generate_lyric("分手", 50)

    print("end")