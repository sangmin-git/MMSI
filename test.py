import argparse
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import SocialDataset, collate_fn
from model import MultimodalBaseline


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
    parser.add_argument('--checkpoint_file', type=str, default='enter_the_path', help='File path for loading checkpoint')
    parser.add_argument('--language_model', type=str, default='bert', choices=['bert', 'roberta', 'electra'], help='Language model to use')
    parser.add_argument('--max_people_num', type=int, default=6, help='Maximum number of total players')
    parser.add_argument('--context_length', type=int, default=5, help='Size of conversation context')
    parser.add_argument('--batch_size', type=int, default=16, help='Mini-batch size')
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

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for language_tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels in dataloader:
            task_labels = task_labels.to(device)
            outputs = model(language_tokens, mask_idxs, keypoint_seqs, speaker_labels, warmup=False)
            _, predicted = torch.max(outputs.data, 1)
            total += task_labels.size(0)
            correct += (predicted == task_labels).sum().item()

    return correct / total

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultimodalBaseline(args.max_people_num, args.language_model).to(device)

    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded checkpoint from {args.checkpoint_file}")

    tokenizer = get_tokenizer(args.language_model)
    args.tokenizer = tokenizer

    collate_fn_token = partial(collate_fn, tokenizer)
    test_dataset = SocialDataset(args, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn_token,
                             shuffle=False, num_workers=args.workers, drop_last=False)

    test_acc = evaluate(model, test_loader, device)

    print(f"Test Accuracy: {test_acc:.3f}")
    print(f"Model: {args.model_name}")

if __name__ == '__main__':
    main()