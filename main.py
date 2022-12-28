import math
import numpy as np
import pandas as pd
import telebot
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import youtokentome as yttm
import torch.nn.functional as F

from tqdm import tqdm

from matplotlib import pyplot as plt

from tokenization import TrainDataset, Tokenizer
from model import Encoder, Decoder, CNN_Seq2Seq


TOKEN = '5444982518:AAEbJo_aXXmSfT8Ylhy2ckG3pViTVYAS7Vk'
bot = telebot.TeleBot(TOKEN)

vocab_size = 30_000

INPUT_DIM = vocab_size
OUTPUT_DIM = vocab_size
KERNEL_SIZE = 2
ENC_EMB_DIM = 300
DEC_EMB_DIM = 300
HID_DIM = 512


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

try:
    model_path = 'pretrained_bpe_lm.model'
except:
    model_path = 'pretrained_bpe_lm.model'
    yttm.BPE.train(data='for_bpe.txt', vocab_size=vocab_size, model=model_path)
tokenizer = Tokenizer(model_path, 64)

new_m = CNN_Seq2Seq(INPUT_DIM, OUTPUT_DIM, ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM, KERNEL_SIZE, device)
new_m.load_state_dict(torch.load('model_seq2seq.pt', map_location=torch.device('cpu')))
new_m.eval()
# model = torch.load('tut5-model.pt', map_location=torch.device('cpu'))


def evaluating(dataloader, model, criterion, optimizer, n_epoch):
    losses = []

    print('Epoch #{}\n'.format(n_epoch + 1))

    model.eval()

    try:

        progress_bar = tqdm(total=len(dataloader.dataset) / 64, desc='Epoch {}'.format(n_epoch + 1))

        for x, y, y_no_eos in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_no_eos = y_no_eos.to(device)

            output, _ = model(x, y_no_eos[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            y = y[:, 1:].contiguous().view(-1)

            loss = criterion(output, y)

            losses.append(loss.item())

            progress_bar.update()
            progress_bar.set_postfix(loss=np.mean(losses[-100:]))

        progress_bar.close()

    except KeyboardInterrupt:

        progress_bar.close()
        # break

    print(f'\tValidation Loss: {np.mean(losses[-100:])}')

    return np.mean(losses[-100:])


def training(dataloader, model, criterion, optimizer, n_epoch, clip):
    losses = []

    print('Epoch #{}\n'.format(n_epoch + 1))

    model.train()

    try:

        progress_bar = tqdm(total=len(dataloader.dataset) / 64, desc='Epoch {}'.format(n_epoch + 1))

        for x, y, y_no_eos in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_no_eos = y_no_eos.to(device)

            optimizer.zero_grad()

            output, _ = model(x, y_no_eos[:, :-1])

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            y = y[:, 1:].contiguous().view(-1)

            loss = criterion(output, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            losses.append(loss.item())

            progress_bar.update()
            progress_bar.set_postfix(loss=np.mean(losses[-100:]))

        progress_bar.close()

    except KeyboardInterrupt:

        progress_bar.close()
        # break

    print(f'\tTrain Loss: {np.mean(losses[-100:])}')

    return np.mean(losses[-100:])


def apply_train():
    dataset = TrainDataset('train.txt')
    train, dev, test = dataset.train, dataset.val, dataset.test
    dataset.prepare_tokenization()
    vocab_size = 30_000

    train_loader = DataLoader(train, batch_size=64, shuffle=True, collate_fn=lambda x: tokenizer.collate(x))
    dev_loader = DataLoader(dev, batch_size=64, shuffle=True, collate_fn=lambda x: tokenizer.collate(x))
    test_loader = DataLoader(test, batch_size=64, shuffle=True, collate_fn=lambda x: tokenizer.collate(x))
    model = CNN_Seq2Seq(INPUT_DIM, OUTPUT_DIM, ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM, KERNEL_SIZE, device)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00001)
    n_epoch = 20
    clip = 0.1

    best_valid_loss = float('inf')

    for epoch in range(n_epoch):
        # print('Epoch #{}\n'.format(n_epoch+1))

        train_loss = training(train_loader, model, criterion, optimizer, epoch, clip)
        valid_loss = evaluating(dev_loader, model, criterion, optimizer, epoch)

    torch.save(model.state_dict(), 'tut5-model.pt')


def generate(sentence, model, bos_index=2, eos_index=3, max_sequence=45):

    tokenized = tokenizer.tokenize([sentence])

    # добавляем тег начала предложения
    tokenized[0].insert(0, bos_index)

    # print(tokenized)
    x = torch.tensor(tokenized).long().to(device)

    model.eval()

    with torch.no_grad():
        encoder_conved, encoder_combined = model.encoder(x)


    trg_indexes = [[bos_index]]

    for i in range(max_sequence):
        trg_tensor = torch.LongTensor(trg_indexes).to(device)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, encoder_conved, encoder_combined)
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes[0].append(pred_token)
        if pred_token == eos_index:
            break

    trg_tokens = tokenizer.tokenizer.decode(trg_indexes[0])

    return ' '.join(trg_tokens)


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id,'Привет')


@bot.message_handler(content_types='text')
def message_reply(message):
    gen_answer = generate(message.text, new_m, bos_index=2, eos_index=3, max_sequence=20).replace('<BOS>', '').replace('==', '')
    bot.send_message(message.chat.id, gen_answer)


def __main__():
    bot.infinity_polling()


__main__()