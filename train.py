import argparse
import random
from functools import partial
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from dataloader import SocialDataset, collate_fn
from model import MultimodalBaseline
from utils import AverageMeter, Progbar


seed = 1234
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='model_name', help='Name of the model')
    parser.add_argument('--task', type=str, default='STI', choices=['STI', 'PCR', 'MPP'], help='Task to perform')
    parser.add_argument('--txt_dir', type=str, default='enter_the_path', help='Directory of anonymized transcripts')
    parser.add_argument('--txt_labeled_dir', type=str, default='enter_the_path', help='Directory of labeled anonymized transcripts')
    parser.add_argument('--keypoint_dir', type=str, default='enter_the_path', help='Directory of keypoints')
    parser.add_argument('--meta_dir', type=str, default='enter_the_path', help='Directory of game meta data')
    parser.add_argument('--data_split_file', type=str, default='enter_the_path', help='File path for data split')
    parser.add_argument('--checkpoint_save_dir', type=str, default='./checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--language_model', type=str, default='bert', choices=['bert', 'roberta', 'electra'], help='Language model to use')
    parser.add_argument('--max_people_num', type=int, default=6, help='Maximum number of total players')
    parser.add_argument('--context_length', type=int, default=5, help='Size of conversation context')
    parser.add_argument('--batch_size', type=int, default=16, help='Mini-batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of total epochs')
    parser.add_argument('--epochs_warmup', type=int, default=10, help='Number of visual warmup epochs')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
    return parser.parse_args()

def get_tokenizer(language_model):
    if language_model == 'bert':
        from transformers import BertTokenizer
        return BertTokenizer.from_pretrained('bert-base-uncased')
    elif language_model == 'roberta':
        from transformers import RobertaTokenizer
        return RobertaTokenizer.from_pretrained('roberta-base')
    elif language_model == 'electra':
        from transformers import ElectraTokenizer
        return ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
    else:
        raise ValueError(f"Unsupported language model: {language_model}")

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch, args):
    model.train()
    train_loss = AverageMeter()
    progbar = Progbar(len(dataloader.dataset))

    for language_tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels in dataloader:
        optimizer.zero_grad()

        task_labels = task_labels.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(language_tokens, mask_idxs, keypoint_seqs, speaker_labels, warmup=(epoch < args.epochs_warmup))
            loss = criterion(outputs, task_labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss.update(loss.item(), task_labels.size(0))
        progbar.add(args.batch_size, values=[('loss', loss.item())])

    return train_loss.avg

def evaluate(model, dataloader, device, epoch, args):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for language_tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels in dataloader:
            task_labels = task_labels.to(device)
            outputs = model(language_tokens, mask_idxs, keypoint_seqs, speaker_labels, warmup=(epoch < args.epochs_warmup))
            _, predicted = torch.max(outputs.data, 1)
            total += task_labels.size(0)
            correct += (predicted == task_labels).sum().item()

    return correct / total

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_save_dir, exist_ok=True)

    model = MultimodalBaseline(args.max_people_num, args.language_model).to(device)

    language_params = [p for n, p in model.named_parameters() if 'convers_encoder' in n]
    other_params = [p for n, p in model.named_parameters() if 'convers_encoder' not in n]
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': args.learning_rate * 10},
        {'params': language_params, 'lr': args.learning_rate}
    ])

    tokenizer = get_tokenizer(args.language_model)
    args.tokenizer = tokenizer

    collate_fn_token = partial(collate_fn, tokenizer)
    train_dataset = SocialDataset(args, is_training=True)
    test_dataset = SocialDataset(args, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn_token,
                              shuffle=True, num_workers=args.workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn_token,
                             shuffle=False, num_workers=args.workers, drop_last=False)

    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_acc = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, args)
        test_acc = evaluate(model, test_loader, device, epoch, args)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save({
                'model_name': args.model_name,
                'model': model.state_dict(),
            }, f"{args.checkpoint_save_dir}/model.pt")

        print()
        print(f"Epoch: {epoch + 1}")
        print(f"Train Loss: {train_loss:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")
        print(f"Test Accuracy (Best): {best_acc:.3f} / {best_epoch + 1}e")
        print(f"Model: {args.model_name}")

if __name__ == '__main__':
    main()
