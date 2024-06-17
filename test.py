from dataloader import *
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
num_workers = 0

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='model_name')
parser.add_argument('--task', type=str, default='STI', help='STI or PCR or MPP')
parser.add_argument('--txt_dir', type=str, default='enter_the_path', help='directory of anonymized transcripts')
parser.add_argument('--txt_labeled_dir', type=str, default='enter_the_path', help='directory of labeled anonymized transcripts')
parser.add_argument('--keypoint_dir', type=str, default='enter_the_path', help='directory of keypoints')
parser.add_argument('--meta_dir', type=str, default='enter_the_path', help='directory of game meta data')
parser.add_argument('--data_split_file', type=str, default='enter_the_path', help='file path for data split')
parser.add_argument('--checkpoint_file', type=str, default='enter_the_path', help='file path for loading checkpoint')
parser.add_argument('--language_model', type=str, default='bert', help='bert or roberta or electra')
parser.add_argument('--max_people_num', type=float, default=6, help='maximum number of total players')
parser.add_argument('--context_length', type=float, default=5, help='size of conversation context')
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
args = parser.parse_args()

def main():
    model = MultimodalBaseline(args.max_people_num, args.language_model).cuda()
    load_state_dict = torch.load(args.checkpoint_file)
    model.load_state_dict(load_state_dict['model'], strict=True)

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
    test_dataset = SocialDataset(args, is_training=False)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn_token, shuffle=False, num_workers=num_workers, drop_last=False)

    correct_num = 0
    sample_num = 0
    model.eval()
    for language_tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels in testloader:
        with torch.no_grad():
            outputs = model(language_tokens, mask_idxs, keypoint_seqs, speaker_labels, warmup=False)

        _, prediction = torch.topk(outputs, k=1, dim=1)
        correct_num += (prediction.squeeze() == task_labels).float().sum()
        sample_num += task_labels.size(0)

    acc = correct_num / sample_num

    print(' * test_acc: {:.3f} '.format(acc))
    print(' * Model name: {}'.format(args.model_name))

if __name__ == '__main__':
    main()
