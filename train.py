import os
import logging
import json
import statistics
import csv
from typing import Optional, Union, List
import pandas as pd
import numpy as np
import shutil
from tqdm import tqdm
from collections import defaultdict, Counter
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # https://github.com/pytorch/pytorch/issues/57273
import pathlib
from pathlib import Path
import argparse

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import transformers # kakao kogpt requires transformers version 4.12.0
from transformers.optimization import get_scheduler

import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

print('pytorch ver:', torch.__version__)
print('transformers ver:', transformers.__version__)

from loss import CrossEntropy, FocalCrossEntropy, label2target, get_loss
from utils import create_logger, create_directory, increment_path, save_performance_graph, Evaluator, get_optimizer
from dataset import preprocess, KOGPT2ClassifyDataset, KOGPT3ClassifyDataset, KOBERTClassifyDataset
from network import KOGPT2Classifier, KOGPT3Classifier, KOBERTClassifier

FILE = Path(__file__).resolve()
DATA = FILE.parents[2]
ROOT = FILE.parents[0]  # root directory
save_dir = increment_path(Path(ROOT) / 'runs'/ 'train' / 'exp')
    
# Dataset
parser=argparse.ArgumentParser(
        description='Training Disease Recognition in Pet CT')
# parser.add_argument('root', metavar='DIR',
#                     help='path to data')
parser.add_argument('--root', default=DATA / 'data' / '1. 실습용자료_hsp2.txt', type=str,
                    help='data format should be txt, sep="|"')
parser.add_argument('--project', default=save_dir, type=str)
parser.add_argument('--num-test', default=100000, type=int,
                    help='the number of test data')
parser.add_argument('--upsample', default='', type=str,
                    help='"shuffle", "reproduce", "random"')
parser.add_argument('--minimum', default=500, type=int,
                    help='(upsample) setting the minimum number of data of each categories')
parser.add_argument('--target', default='S', type=str,
                    help='target')
# parser.add_argument('--num_test_ratio', default=0.1, type=float,
#                     help='a ratio of test data')

# DataLoader
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N',
                    help='mini-batch size (default: 16)'
                         '[kobert] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50'
                         '[kogpt2] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50'
                         '[kogpt3] a NVDIA RTX 3090T memory can process 512 batch size where max_len is 50')

# Model
parser.add_argument('-m', '--model', default='kobert', type=str,
                    help='Model to train. Available models are ["kobert", "kogpt2", "kogpt3"]. default is "kogpt3".')
parser.add_argument('--dr-rate', default=None, type=float,
                    help='')
parser.add_argument('--bias-off', action='store_false',
                    help='')

# Train setting
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--patience', default=10, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--additional-train', action='store_true',
                    help='additional train')  
parser.add_argument('--additional-epochs', default=5, type=int, metavar='N',
                    help='additional train epochs')              
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

# Loss
parser.add_argument('--loss', default='FCE', type=str,
                    help='Loss function. Availabel loss functions are . default is Focal Cross Entropy(FCE).')

# Learning rate
parser.add_argument('-lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-scheduler', default='cosine_with_restarts',
                    type=str, help='Available schedulers are "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup')
parser.add_argument('--warmup-ratio', default=0.01, type=float, help='lr-scheduler')


# Optimizer
parser.add_argument('--optimizer', type=str, default='AdamW',
                    help='default is AdamW')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='momentum1 in Adam')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='momentum2 in Adam')
parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight_decay')
parser.add_argument('-eps', '--epsilon', type=float, default=1e-8)
parser.add_argument('--amsgrad', action='store_true')

# Single GPU Train
parser.add_argument('--device', default='cuda', type=str,
                    help='device to use. "cpu", "cuda", "cuda:0", "cuda:1"')

parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training.')
parser.add_argument('--max-len', default=50, type=int,
                    help='max sequence length to cut or pad')


args=parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(torch.cuda.device_count())))
os.environ["CUDA_LAUNCH_BLOCKING"] = ",".join(map(str, range(torch.cuda.device_count())))

create_directory(args.project / 'weights')
create_directory(args.project)
logger = create_logger(args.project, file_name='log.txt')

# save config
with open(args.project / 'config.json', 'w', encoding='cp949') as f:
    arg_dict = {k: (str(v) if type(v)==pathlib.PosixPath else v) for k, v in args.__dict__.items()}
    json.dump(arg_dict, f, indent=4)

print('output path:', args.project)

best_acc = None
best_loss = None

def main(args):
    global best_acc
    global best_loss
    
    # preprocess data
    (model, train_set, test_set), cat2id, id2cat = get_model_dataset(args.model, args.dr_rate, args.bias_off, args.root, args.num_test, args.upsample, args.minimum, args.target, args.max_len, args.seed)
    model = model.to(args.device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=True, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers,
                             shuffle=False, pin_memory=False)
    
    logger.info(f'# train data: {len(train_set)}')
    logger.info(f'# test  data: {len(test_set)}')
    
    with open(args.project / 'cat2id.json', 'w', encoding='cp949') as f:
        json.dump(cat2id, f, indent=4)
    with open(args.project / 'id2cat.json', 'w', encoding='cp949') as f:
        json.dump(id2cat, f, indent=4)
    
    # optimizer
    betas=(args.beta1, args.beta2)
    optimizer = get_optimizer(optimizer_type=args.optimizer, model=model, lr=args.lr, betas=betas,
                              weight_decay=args.weight_decay, eps=args.epsilon, amsgrad=args.amsgrad)
    
    # lr-scheduler
    max_iter = len(train_loader) * args.epochs
    num_warmup_steps=int(max_iter * args.warmup_ratio)
    scheduler = get_scheduler(name=args.lr_scheduler, optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_iter)
                    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.device is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint = torch.load(args.resume, map_location=args.device)
            # build model
            model.load_state_dict(checkpoint['state_dict'])
            # build optimizer
            optimizer.load_state_dict(checkpoint['optimizer'])
            # build scheduler
            scheduler.load_state_dict(checkpoint['scheduler'])
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['acc']
            best_acc = checkpoint['loss']
            
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            print('start epoch: {}'.format(args.start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # loss function
    criterion = get_loss(args.loss)
            
    # train
    for epoch in range(args.start_epoch, args.epochs):
        # epoch train
        eval_train = train(model, train_loader, optimizer, criterion, scheduler, args.device)
    
        # epoch validation
        eval_valid = valid(model, test_loader, criterion, args.device)
        
        # logging scores
        logger.info(f'Epoch {epoch} Result')
        logger.info(f'\ttrain | loss: {eval_train.loss}\tacc: {round(eval_train.acc, 6)}\tpc: {round(eval_train.macro_pc, 6)}\trc: {round(eval_train.macro_rc, 6)}\tf1: {round(eval_train.macro_f1, 6)}')
        logger.info(f'\tvalid | loss: {eval_valid.loss}\tacc: {round(eval_valid.acc, 6)}\tpc: {round(eval_valid.macro_pc, 6)}\trc: {round(eval_valid.macro_rc, 6)}\tf1: {round(eval_valid.macro_f1, 6)}')
        
        # save scores
        if epoch==args.start_epoch:
            # summary.csv
            with open(args.project / 'summary.csv', 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(['epoch', 'train loss', 'train acc', 'train pc', 'train rc', 'train f1',
                                          'valid loss', 'valid acc', 'valid pc', 'valid rc', 'valid f1'])
            # base frame for precisions, recalls and f1scores
            class_id = list(set(train_loader.dataset.label))
            num_train_data, num_valid_data = [0] * len(class_id), [0] * len(class_id)
            for c_id, n in dict(Counter(train_loader.dataset.label)).items():
                num_train_data[c_id] = n
            for c_id, n in dict(Counter(test_loader.dataset.label)).items():
                num_valid_data[c_id] = n
            history_train = defaultdict(lambda: pd.DataFrame({
                    'class_id': class_id,
                    'class': list(map(lambda x: ''.join(id2cat[x]), class_id)),
                    '# train data' : num_train_data,
                    '# valid data' : num_valid_data
                }))
            history_valid = defaultdict(lambda: pd.DataFrame({
                'class_id': class_id,
                'class': list(map(lambda x: ''.join(id2cat[x]), class_id)),
                '# train data' : num_train_data,
                '# valid data' : num_valid_data
            }))
            
        # add new line to summary.csv
        with open(args.project / 'summary.csv', 'a', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([epoch, eval_train.loss, eval_train.acc, eval_train.macro_pc, eval_train.macro_rc, eval_train.macro_f1,
                                    eval_valid.loss, eval_valid.acc, eval_valid.macro_pc, eval_valid.macro_rc, eval_valid.macro_f1])
            
        # add new column(epoch) to precision.csv, recall.csv and f1score.csv
        for metric, values in eval_train.class_scores.items():
            if metric != 'class_id':
                history_train[metric][f'epoch {epoch}'] = 0
                for c_id, v in zip(eval_valid.class_scores['class_id'], values):
                    r = history_train[metric][history_train[metric]['class_id']==c_id][f'epoch {epoch}'].index
                    history_train[metric].loc[r, f'epoch {epoch}'] = v
                history_train[metric].to_csv(args.project / f'{metric}_train.csv', encoding='utf-8-sig', index=False)

        # add new column(epoch) to precision.csv, recall.csv and f1score.csv
        for metric, values in eval_valid.class_scores.items():
            if metric != 'class_id':
                history_valid[metric][f'epoch {epoch}'] = 0
                for c_id, v in zip(eval_valid.class_scores['class_id'], values):
                    r = history_valid[metric][history_valid[metric]['class_id']==c_id][f'epoch {epoch}'].index
                    history_valid[metric].loc[r, f'epoch {epoch}'] = v
                history_valid[metric].to_csv(args.project / f'{metric}_valid.csv', encoding='utf-8-sig', index=False)
            
        # save performance graph
        save_performance_graph(args.project / 'summary.csv', args.project / 'performance.png')
        
        # model save
        torch.save({'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best acc': eval_valid.acc if best_acc is None or eval_valid.acc > best_acc else best_acc,
                    'best loss': eval_valid.loss if best_loss is None or valid_loss < best_loss else eval_valid.loss,
                    'epoch': epoch,
                    },
                   args.project / 'weights' / 'checkpoint.pth.tar')
        
        if  best_acc is None or eval_valid.acc > best_acc: 
            print(f'Validation acc got better {best_acc} --> {eval_valid.acc}.  Saving model ...')
            shutil.copyfile(args.project / 'weights' / 'checkpoint.pth.tar', args.project / 'weights' / 'best_acc.pth.tar')
            best_acc = eval_valid.acc
            print(len(test_loader.dataset.doc), len(test_loader.dataset.label), len(eval_valid.predictions))
            # save valid predictions
#             pred_frame = pd.DataFrame({
#                 "doc": test_loader.dataset.doc,
#                 "category": list(map(lambda x: ''.join(id2cat[x]), test_loader.dataset.label)),
#                 "predictions": list(map(lambda x: ''.join(id2cat[x]), eval_valid.predictions))
#             })
#             pred_frame.to_csv(args.project / 'best_acc_predictions.csv', encoding='utf-8-sig', index=False)
        
        if  best_loss is None or eval_valid.loss < best_loss: 
            print(f'Validation loss got better {best_loss} --> {eval_valid.vloss}.  Saving model ...')
            shutil.copyfile(args.project / 'weights' / 'checkpoint.pth.tar', args.project / 'weights' / 'best_loss.pth.tar')
            best_loss = eval_valid.loss
            
#             # save valid predictions
#             pred_frame = pd.DataFrame({
#                 "doc": test_loader.dataset.doc,
#                 "category": list(map(lambda x: ''.join(id2cat[x]), test_loader.dataset.label)),
#                 "predictions": list(map(lambda x: ''.join(id2cat[x]), eval_valid.predictions))
#             })
#             pred_frame.to_csv(args.project / 'best_loss_predictions.csv', encoding='utf-8-sig', index=False)
            patience = 0
        else:
            logger.info(f'patience {patience} --> {patience+1}')
            patience += 1
        
        if patience >= args.patience:
            logger.info('Early Stop!')
            break
            
    # additional training
    if args.additional_train:
        logger.info(f'Additional Training with Validation Dataset for {args.additional_epochs} epochs')

        checkpoint = torch.load(args.project / 'weights' / 'best_loss.pth.tar', map_location=args.device)
        model.load_state_dict(checkpoint['state_dict']) # build model
        logger.info('load model')
        betas=(args.beta1, args.beta2)
        optimizer = get_optimizer(optimizer_type=args.optimizer, model=model, lr=args.lr, betas=betas,
                                  weight_decay=args.weight_decay, eps=args.epsilon, amsgrad=args.amsgrad)
        # lr-scheduler
        max_iter = len(train_loader) * args.epochs
        num_warmup_steps = int(args.warmup_ratio * max_iter)
        scheduler = get_scheduler(name=args.lr_scheduler, optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_iter)
        logger.info(f'max_iteration: {max_iter}')
        logger.info(f'num_warmup_step: {num_warmup_steps}')
    #     optimizer.load_state_dict(checkpoint['optimizer']) # build optimizer
    #     logger.info('load optimizer')
    #     scheduler.load_state_dict(checkpoint['scheduler']) # build scheduler
    #     logger.info('load scheduler')
        ad_train_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers,
                                 shuffle=True, pin_memory=False)
        ad_test_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                                 shuffle=False, pin_memory=False)

        logger.info('start add-train')
        for epoch in range(args.additional_epochs):
            # epoch train
            eval_train = train(model, ad_train_loader, optimizer, criterion, scheduler, args.device)
            eval_valid = valid(model, ad_test_loader, criterion, args.device)

            # logging scores
            logger.info(f'Epoch {epoch} Result')
            logger.info(f'\ttrain | loss: {eval_train.loss}\tacc: {round(eval_train.acc, 6)}\tpc: {round(eval_train.macro_pc, 6)}\trc: {round(eval_train.macro_rc, 6)}\tf1: {round(eval_train.macro_f1, 6)}')
            logger.info(f'\tvalid | loss: {eval_valid.loss}\tacc: {round(eval_valid.acc, 6)}\tpc: {round(eval_valid.macro_pc, 6)}\trc: {round(eval_valid.macro_rc, 6)}\tf1: {round(eval_valid.macro_f1, 6)}')

        torch.save({'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': checkpoint['epoch'],
                    'additional_epoch': epoch
                   },
                    args.project / 'weights' / 'additional.pth.tar')
        logger.info('save model')
        
def train(model, train_loader, optimizer, criterion, scheduler, device):
    eval_train = Evaluator(model.num_classes)
    model.train()
    
#     with torch.autograd.detect_anomaly():
    for (input_ids, attention_mask, token_type_ids, label) in tqdm(train_loader, total=len(train_loader)):
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        token_type_ids = token_type_ids.to(device, non_blocking=True)

        # forward propagation
        output = model(input_ids, attention_mask, token_type_ids)
        target = label2target(output, label).to(device, non_blocking=True)
        loss = criterion(output, target)
        
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        
        # update score
        pred = output.argmax(1)
        eval_train.update(pred.tolist(), label.tolist(), loss=float(loss)*len(label))
    eval_train.compute()
    return eval_train
        
    
def valid(model, valid_loader, criterion, device):
    eval_valid = Evaluator(model.num_classes)
    
    model.eval()
    with torch.no_grad():
        for (input_ids, attention_mask, token_type_ids, label) in tqdm(valid_loader, total=len(valid_loader)):
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            token_type_ids = token_type_ids.to(device, non_blocking=True)

            # forward propagation
            output = model(input_ids, attention_mask, token_type_ids)
            target = label2target(output, label).to(device, non_blocking=True)
            loss = criterion(output, target)
            
            # update score
            pred = output.argmax(1)
            eval_valid.update(pred.tolist(), label.tolist(), loss=float(loss)*len(label))
    eval_valid.compute()
            
    return eval_valid


def get_model_dataset(model_type, dr_rate, bias, root, num_test, upsample, minimum, target, max_len, seed):
    def _get_kobert_model_dataset(num_classes, dr_rate, bias, doc_train, label_train, doc_test, label_test, max_len):
        kobert, vocab = get_pytorch_kobert_model()
        tokenizer_path = get_tokenizer()
        tokenizer = nlp.data.BERTSPTokenizer(tokenizer_path, vocab, lower=False)
        transform = nlp.data.BERTSentenceTransform(
                    tokenizer, max_seq_length=max_len, pad=True, pair=False) 
        
        train_set = KOBERTClassifyDataset(doc_train, label_train, transform)
        test_set = KOBERTClassifyDataset(doc_test, label_test, transform)
        
        model = KOBERTClassifier(kobert, num_classes=num_classes, dr_rate=dr_rate, bias=bias)
        return model, train_set, test_set
    
    def _get_kogpt2_model_dataset(num_classes, dr_rate, bias, doc_train, label_train, doc_test, label_test, max_len):
        train_set = KOGPT2ClassifyDataset(doc_train, label_train, max_len=max_len, padding='max_length', truncation=True)
        test_set = KOGPT2ClassifyDataset(doc_test, label_test, max_len=max_len, padding='max_length', truncation=True)
        
        model = KOGPT2Classifier(num_classes=num_classes, pad_token_id = train_set.tokenizer.eos_token_id, dr_rate=dr_rate, bias=bias)
        return model, train_set, test_set
    
    def _get_kogpt3_model_dataset(num_classes, dr_rate, bias, doc_train, label_train, doc_test, label_test, max_len):
        train_set = KOGPT3ClassifyDataset(doc_train, label_train, max_len=max_len, padding='max_length', truncation=True)
        test_set = KOGPT3ClassifyDataset(doc_test, label_test, max_len=max_len, padding='max_length', truncation=True)
        
        model = KOGPT3Classifier(num_classes=num_classes, pad_token_id = train_set.tokenizer.eos_token_id, dr_rate=dr_rate, bias=bias)
        return model, train_set, test_set
    
    try:
        data = pd.read_csv(root, sep='|', encoding='euc-kr')
    except:
        data = pd.read_csv(root, sep='|', encoding='utf-8')
        
    train, test, cat2id, id2cat = preprocess(data, num_test=num_test, upsample=upsample, minimum=minimum, target=target, seed=seed)
    doc_train, doc_test, label_train, label_test = train['text'].tolist(), test['text'].tolist(), train['label'].tolist(), test['label'].tolist()
    num_classes = len(cat2id.keys())
    
    if model_type=='kobert':
        return _get_kobert_model_dataset(num_classes, dr_rate, bias, doc_train, label_train, doc_test, label_test, max_len), cat2id, id2cat
    elif model_type=='kogpt2':
        return _get_kogpt2_model_dataset(num_classes, dr_rate, bias, doc_train, label_train, doc_test, label_test, max_len), cat2id, id2cat
    elif model_type=='kogpt3':
        return _get_kogpt3_model_dataset(num_classes, dr_rate, bias, doc_train, label_train, doc_test, label_test, max_len), cat2id, id2cat
    else:
        raise
        

if __name__=='__main__':
    main(args)