# -*- coding: utf-8 -*-
# @Time    : 2021/1/15 15:22
# @Author  : kaka

import os
import torch
import torch.nn.functional as F
import json
from model.textcnn import TextCNN
import dataset


def train_model(train_iter, val_iter, model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    step_idx = 0
    best_acc = 0
    best_step = 0
    model.train()
    for epoch in range(1, args['epochs'] + 1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            # feature = feature.data.t_()
            feature = feature.transpose(0, 1)
            optimizer.zero_grad()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            step_idx += 1
            if step_idx % args['display_interval'] == 0:
                corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
                train_acc = 100.0 * corrects / batch.batch_size
                print('Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(step_idx,
                                                                             loss.item(),
                                                                             train_acc,
                                                                             corrects,
                                                                             batch.batch_size))
            if step_idx % args['val_interval'] == 0:
                val_acc = eval_model(val_iter, model, args)
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_step = step_idx
                    print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                    save(model, './', best_acc, step_idx)
                else:
                    if step_idx - best_step >= args['early_stopping']:
                        print('early stop by {} steps, acc: {:.4f}%'.format(args['early_stopping'], best_acc))
                        raise KeyboardInterrupt


def eval_model(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        # feature.data.t_()
        feature = feature.transpose(0, 1)
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


def save(model, save_dir, acc, step):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = '{0}/step{1}_acc{2:.4f}.pt'.format(save_dir, step, acc)
    torch.save(model.state_dict(), save_path)


def main():
    with open('./conf/textcnn_conf.json', 'r') as h_in:
        args = json.load(h_in)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iter, val_iter, test_iter = dataset.get_data('./data', batch_size=args['batch_size'], device=device)

    vocab_size = len(train_iter.dataset.fields['text'].vocab.stoi)
    model = TextCNN(
        vocab_size=vocab_size,
        class_num=args['class_num'],
        embed_size=args['embed_size'],
        kernel_num=args['kernel_num'],
        kernel_sizes=args['kernel_sizes'],
        dropout_rate=args['dropout_rate']
    )
    print(model)

    train_model(train_iter=train_iter, val_iter=val_iter, model=model, args=args)


if __name__ == '__main__':
    main()
