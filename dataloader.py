import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random
import re
import json
import numpy as np
import copy

def collate_fn(tokenizer, batch):
    tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels = zip(*batch)
    tokens = [torch.tensor(t) for t in tokens]
    tokens = pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id).cuda()
    mask_idxs = torch.tensor(mask_idxs).cuda()
    task_labels = torch.tensor(task_labels).cuda()
    keypoint_seqs = torch.tensor(np.array(keypoint_seqs)).cuda().float()
    speaker_labels = torch.tensor(speaker_labels).cuda()

    return tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels

class SocialDataset(Dataset):
    def __init__(self, args, is_training=True):
        self.is_training = is_training
        self.tokenizer = args.tokenizer
        self.language_model = args.language_model
        self.context_length = args.context_length
        self.txt_labeled_dir = args.txt_labeled_dir
        self.meta_dir = args.meta_dir
        self.keypoint_dir = args.keypoint_dir
        self.task = args.task

        with open(args.data_split_file, 'r') as f:
            data_split = json.load(f)
        self.file_names = self.get_file_names(args, data_split, is_training)
        self.mask_token = self.get_mask_token(self.language_model)
        self.data_points = self.load_files(self.file_names)

    def get_mask_token(self, language_model):
        if language_model in ['bert', 'electra']:
            return '[MASK]'
        if language_model == 'roberta':
            return '<mask>'
    def get_file_names(self, args, data_split, is_training):
        data_type = 'train' if is_training else 'test'
        return [f"{args.txt_dir}/{file_name}.txt" for file_name in data_split[data_type]]

    def load_files(self, txt_files):
        data_points = []

        with open(f"{self.keypoint_dir}/reference_timestamps.json", 'r') as f:
            reference_timestamps = json.load(f)

        for txt_file in txt_files:
            file_name = txt_file.split('/')[-1].split('.txt')[0]
            txt_file_labeled = f"{self.txt_labeled_dir}/{file_name}.txt"
            keypoint_file = f"{self.keypoint_dir}/{file_name}.npy"
            meta_file = f"{self.meta_dir}/{file_name.split('_')[0]}.json"

            with open(txt_file, 'r') as f:
                content = f.read()
            utterances = content.split('\n')
            utterances = [utterance for utterance in utterances if utterance]

            with open(txt_file_labeled, 'r') as f:
                content = f.read()
            utterances_labeled = content.split('\n')
            utterances_labeled = [utterance_labeled for utterance_labeled in utterances_labeled if utterance_labeled]

            keypoint_data = np.load(keypoint_file, allow_pickle=True)

            with open(meta_file, 'r') as f:
                meta_data = json.load(f)
            player_num = len(meta_data[file_name.split('_')[1]]['playerNames'])

            # reference position
            reference_timestamp = reference_timestamps[file_name]
            keypoint_data_ref = keypoint_data[reference_timestamp]
            keypoint_seq_ref = np.zeros((6, 17 * 2))
            for keypoint_ref_i in keypoint_data_ref:
                if keypoint_ref_i['idx'] < player_num:
                    keypoint_wo_conf = np.delete(keypoint_ref_i['keypoints'], np.arange(2, len(keypoint_ref_i['keypoints']), 3))
                    keypoint_seq_ref[keypoint_ref_i['idx'], :] = keypoint_wo_conf[0:17 * 2]

            for utterance_i, utterance in enumerate(utterances):
                time_str = utterance.split(': ')[0].split('(')[1][:-1]
                m, s = map(int, time_str.split(':'))
                time_sec = m * 60 + s

                words = utterance.split()

                # adjust start and end indexes based on the available utterances
                start_idx = max(0, utterance_i-self.context_length-max(0,self.context_length-(len(utterances) - utterance_i)+1))
                end_idx = start_idx + 2*self.context_length + 1

                # check whether the speaker is a player
                is_player_speaker = words[0].startswith('[Player')
                if is_player_speaker:
                    speaker_label = int(re.search(r'\[Player(\d+)\]', words[0]).group(1))

                utterance_involved = False
                for word_i, word in enumerate(words):
                    add_data_point =False

                    # speaking target identification data loader
                    second_pronouns = ['you', 'your']
                    if self.task=='STI' and (word.lower() in second_pronouns) and utterances_labeled[utterance_i].rstrip().endswith(']') \
                            and word_i!=0 and is_player_speaker and not utterance_involved:

                        pronoun_sentence = utterances_labeled[utterance_i]

                        brackets = re.findall(r"\[(.*?)\]", pronoun_sentence)
                        names_in_bracket = [name in [f'Player{utterance_i}' for utterance_i in range(player_num)] for name
                                            in brackets[-1].split()]
                        if names_in_bracket.count(True) == 1:
                            utterance_involved = True
                            words_cp = copy.deepcopy(words)
                            task_label = int(re.search(r'Player(\d+)', brackets[-1]).group(1))
                            words_cp.append('(To ' + self.mask_token + ')')
                            add_data_point = True

                    # pronoun coreference resolution data loader
                    third_pronouns = ['he', 'his', 'him', 'she', 'her']
                    if self.task=='PCR' and any(pronoun in word.lower() for pronoun in third_pronouns) and \
                            ('Player' in utterances_labeled[utterance_i].split()[word_i]) and word_i!=0 and is_player_speaker:
                        words_cp = copy.deepcopy(words)
                        task_label = int(re.search(r'Player(\d+)', utterances_labeled[utterance_i].split()[word_i]).group(1))
                        target_pronoun = [pronoun for pronoun in third_pronouns if pronoun in word.lower()][-1] #find she first than he
                        words_cp[word_i] = re.sub(rf'{target_pronoun}', self.mask_token, words_cp[word_i].lower())
                        add_data_point = True

                    # mentioned player prediction data loader
                    if self.task=='MPP' and word.startswith('[Player') and word_i!=0 and is_player_speaker:
                        words_cp = copy.deepcopy(words)
                        task_label = int(re.search(r'\[Player(\d+)\]', words_cp[word_i]).group(1))
                        words_cp[word_i] = re.sub(r'\[Player\d\]', self.mask_token, words_cp[word_i])
                        add_data_point = True

                    if add_data_point:
                        # extract keypoints
                        keypoint_seq = np.zeros((6, 16, 17 * 2))
                        for time_i in range(16):
                            time_step = min(max(0, 5*(time_sec-1)+time_i), len(keypoint_data)-1)
                            keypoints = keypoint_data[time_step]
                            for keypoint_i in keypoints:
                                if keypoint_i['idx'] < player_num:
                                    keypoint_wo_conf = np.delete(keypoint_i['keypoints'], np.arange(2, len(keypoint_i['keypoints']), 3))
                                    keypoint_seq[keypoint_i['idx'], time_i, :] = keypoint_wo_conf[0:17 * 2]

                            # position correction for missing players
                            for player_i in range(player_num):
                                if sum(keypoint_seq[player_i,time_i,:]) == 0:
                                    keypoint_seq[player_i,time_i,:] = keypoint_seq_ref[player_i]

                        # normalize keypoints based on the speaker
                        zero_indices = np.where(keypoint_seq == 0)
                        keypoint_seq = keypoint_seq - np.tile(keypoint_seq[speaker_label:speaker_label+1][:,:,0:2],(1,1,17))
                        keypoint_seq[zero_indices] = 0

                        # Extract the conversation context of the target utterance
                        target_utterance = ' '.join(words_cp)
                        utterances_cp = copy.deepcopy(utterances)
                        utterances_cp[utterance_i] = target_utterance
                        convers_context = utterances_cp[start_idx:end_idx]
                        convers_context = [re.sub(r' \(\d{2}:\d{2}\)', '', utterance) for utterance in convers_context] # Remove timestamps

                        data_points.append([convers_context, task_label, player_num, keypoint_seq, speaker_label])

        return data_points

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        convers_context, task_label, player_num, keypoint_seq, speaker_label = self.data_points[idx]

        if self.is_training:
            # flip keypoint
            if random.random() < 0.5:
                keypoint_seq[:,:,::2] = -1.0 * keypoint_seq[:,:,::2]
                keypoint_seq_cp = copy.deepcopy(keypoint_seq)
                for change_i in [1, 3, 5, 7, 9]:
                    keypoint_seq[:, :, 2 * change_i:2 * change_i+1] = keypoint_seq_cp[:, :, 2 * change_i+1:2 * change_i+2]
                    keypoint_seq[:, :, 2 * change_i+1:2 * change_i+2] = keypoint_seq_cp[:, :, 2 * change_i:2 * change_i+1]

            # shuffle player numbers
            player_numbers = list(range(0, player_num))
            shuffled_player_numbers = copy.deepcopy(player_numbers)
            random.shuffle(shuffled_player_numbers)
            player_number_mapping = {old: new for old, new in zip(player_numbers, shuffled_player_numbers)}

            # update the player numbers in the conversation context and labels
            convers_context = [re.sub(r'\[Player(\d+)\]', lambda match: '[Player{}]'.format(player_number_mapping[int(match.group(1))]), utterance) for utterance in convers_context]
            task_label = player_number_mapping[task_label]
            speaker_label = player_number_mapping[speaker_label]

            while len(shuffled_player_numbers) != 6:
                shuffled_player_numbers.append(len(shuffled_player_numbers))

            inverse_shuffle = np.argsort(shuffled_player_numbers).tolist()
            keypoint_seq = keypoint_seq[inverse_shuffle,:,:]
        else:
            shuffled_player_numbers = list(range(0, player_num))
            while len(shuffled_player_numbers) != 6:
                shuffled_player_numbers.append(len(shuffled_player_numbers))

        # tokenize conversation context
        middle_index = len(convers_context) // 2
        tokens = self.tokenizer.encode(convers_context[middle_index], add_special_tokens=False) + [self.tokenizer.sep_token_id]
        i = 0
        while True:
            before_exists = middle_index - i - 1 >= 0
            after_exists = middle_index + i + 1 < len(convers_context)
            before_tokens = self.tokenizer.encode(convers_context[middle_index - i - 1], add_special_tokens=False) if before_exists else []
            after_tokens = self.tokenizer.encode(convers_context[middle_index + i + 1], add_special_tokens=False) if after_exists else []
            if before_exists and len(tokens) + len(before_tokens) + 1 <= 511:  # add sentence before if available
                tokens = before_tokens + [self.tokenizer.sep_token_id] + tokens
            if after_exists and len(tokens) + len(after_tokens) + 1 <= 511:  # add sentence after if available
                tokens = tokens + after_tokens + [self.tokenizer.sep_token_id]
            if not before_exists and not after_exists:
                break
            i += 1

        # find mask indexes
        tokens = [self.tokenizer.cls_token_id] + tokens
        if self.language_model in ['bert', 'electra']:
            mask_token_id = 103
        if self.language_model == 'roberta':
            mask_token_id = 50264

        mask_idx = tokens.index(mask_token_id)

        return tokens, mask_idx, keypoint_seq, speaker_label, task_label
