from dataloader import *
from utils import *
from model import *
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
import random
from functools import partial

scaler = torch.cuda.amp.GradScaler()

seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
num_workers =0

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='model_name')
parser.add_argument('--task', type=str, default='STI', help='STI or PCR or MPP')
parser.add_argument('--txt_dir', type=str, default='enter_the_path', help='directory of anonymized transcripts')
parser.add_argument('--txt_labeled_dir', type=str, default='enter_the_path', help='directory of labeled anonymized transcripts')
parser.add_argument('--keypoint_dir', type=str, default='enter_the_path', help='directory of keypoints')
parser.add_argument('--meta_dir', type=str, default='enter_the_path', help='directory of game meta data')
parser.add_argument('--data_split_file', type=str, default='enter_the_path', help='file path for data split')
parser.add_argument('--checkpoint_save_dir', type=str, default='./checkpoints', help='directory for saving checkpoints')
parser.add_argument('--language_model', type=str, default='bert', help='bert or roberta or electra')
parser.add_argument('--max_people_num', type=float, default=6, help='maximum number of total players')
parser.add_argument('--context_length', type=float, default=5, help='size of conversation context')
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
parser.add_argument('--learning_rate', type=float, default=0.000005, help='learning rate')
parser.add_argument('--epochs', type=float, default=200, help='number of total epochs')
parser.add_argument('--epochs_warmup', type=float, default=10, help='number of visual warmup epochs')
args = parser.parse_args()

def main():
    model = MultimodalBaseline(args.max_people_num, args.language_model).cuda()

    param_name_all = []
    language_params = []
    other_params = []
    for name, param in model.named_parameters():
        param_name_all.append(name)
        if 'language_encoder' in name:
            language_params.append(param)
        else:
            other_params.append(param)

    params = [{'params': other_params, 'lr': args.learning_rate*10},
              {'params': language_params, 'lr': args.learning_rate}]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    if args.language_model == 'bert':
        from transformers import BertTokenizer
        args.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if args.language_model == 'roberta':
        from transformers import RobertaTokenizer
        args.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    if args.language_model == 'electra':
        from transformers import ElectraTokenizer
        args.tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")

    collate_fn_token = partial(collate_fn, args.tokenizer)
    train_dataset = SocialDataset(args, is_training=True)
    test_dataset = SocialDataset(args, is_training=False)

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn_token, shuffle=True, num_workers=num_workers, drop_last=True)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn_token, shuffle=False, num_workers=num_workers, drop_last=False)

    criterion = torch.nn.CrossEntropyLoss()
    train_loss = AverageMeter()
    test_loss = AverageMeter()
    best_acc = 0
    best_epoch_i = 1

    for epoch_i in range(args.epochs):
        progbar = Progbar(len(trainloader.dataset))
        model.train()
        for language_tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels in trainloader:
            optimizer.zero_grad()

            task_labels = task_labels.cuda()
            with torch.cuda.amp.autocast():
                if epoch_i < args.epochs_warmup:
                    outputs = model(language_tokens, mask_idxs, keypoint_seqs, speaker_labels, warmup=True)
                else:
                    outputs = model(language_tokens, mask_idxs, keypoint_seqs, speaker_labels, warmup=False)
                loss = criterion(outputs, task_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss.update(float(loss))
            progbar.add(args.batch_size, values=[('loss', loss.item())])

        print(" // Epoch: {}, Loss: {:.3f}, LR: {}".format(epoch_i+1, train_loss.avg, optimizer.param_groups[0]['lr']))
        train_loss.reset()

        torch.save({'model_name': args.model_name,
                    'model': model.state_dict()},
                   args.checkpoint_save_dir + '/trained_file_' + str(epoch_i + 1).zfill(3) + '.pt')

        model.eval()
        correct_num = 0
        sample_num = 0
        for language_tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels in testloader:
            with torch.no_grad():
                if epoch_i < args.epochs_warmup:
                    outputs = model(language_tokens, mask_idxs, keypoint_seqs, speaker_labels, warmup=True)
                else:
                    outputs = model(language_tokens, mask_idxs, keypoint_seqs, speaker_labels, warmup=False)
                loss = criterion(outputs, task_labels)

            _, prediction = torch.topk(outputs, k=1, dim=1)
            correct_num += (prediction.squeeze() == task_labels).float().sum()
            sample_num += task_labels.size(0)
            test_loss.update(float(loss))

        acc = correct_num / sample_num
        best_acc = max(acc, best_acc)
        if acc >= best_acc:
            best_epoch_i = epoch_i

        print(' * Epoch: {} / test_loss: {:.3f}'.format(epoch_i+1, test_loss.avg))
        print(' * test_acc: {:.3f} '.format(acc))
        print(' * Best test_acc: {:.3f} / {}e '.format(best_acc, best_epoch_i+1))
        print(' * Model name: {}'.format(args.model_name))
        test_loss.reset()

if __name__ == '__main__':
    main()
