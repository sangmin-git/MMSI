## Modeling Multimodal Social Interactions: New Challenges and Baselines with Densely Aligned Representations


<div align="center"><img width="98%" src="https://github.com/sangmin-git/MMSI/raw/main/assets/figure.png" /></div>


>
This repository contains the official PyTorch implementation of the following paper:
> **Modeling Multimodal Social Interactions: New Challenges and Baselines with Densely Aligned Representations (CVPR 2024 Oral)**<br>
> Sangmin Lee, Bolin Lai, Fiona Ryan, Bikram Boote, James M. Rehg <br>
> Paper: https://arxiv.org/abs/2403.02090<br>
>
> **Abstract** *Understanding social interactions involving both verbal and non-verbal cues is essential for effectively interpreting social situations. However, most prior works on multimodal social cues focus predominantly on single-person behaviors or rely on holistic visual representations that are not aligned to utterances in multi-party environments. Consequently, they are limited in modeling the intricate dynamics of multi-party interactions. In this paper, we introduce three new challenging tasks to model the fine-grained dynamics between multiple people: speaking target identification, pronoun coreference resolution, and mentioned player prediction. We contribute extensive data annotations to curate these new challenges in social deduction game settings. Furthermore, we propose a novel multimodal baseline that leverages densely aligned language-visual representations by synchronizing visual features with their corresponding utterances. This facilitates concurrently capturing verbal and non-verbal cues pertinent to social reasoning. Experiments demonstrate the effectiveness of the proposed approach with densely aligned multimodal representations in modeling fine-grained social interactions.*

## Preparation

### Requirements
- python 3
- pytorch 2.0+
- transformers
- numpy

### Datasets
- Download the benchmark datasets (YouTube, Ego4D) from [[link](https://www.dropbox.com/scl/fo/fbv6njzu1ynbgv9wgtrwo/ANPk2TKqK2rl44MqKu05ogk?rlkey=yx7bmzmmiymauvz99q2rvjajg&st=305631zj&dl=0)]
- For access to the original datasets including videos, visit [[link](https://persuasion-deductiongame.socialai-data.org/)].
- You can download the aligned player keypoint samples from [[link](https://www.dropbox.com/scl/fo/01rp8c126kc9014kbhvkg/AO2JvbsFuMd4WkkwzzOR06U?rlkey=910f1sf90zm6piii0krepikzi&st=u36zodh8&dl=0)]

## Training
`train.py` saves the weights in `--checkpoint_save_dir` and shows the training logs.

To train the model, run following command:
```shell
# Training example for speaking target identification
python train.py \
--task 'STI' \
--txt_dir 'enter_the_path' --txt_labeled_dir 'enter_the_path' \
--keypoint_dir 'enter_the_path' --meta_dir 'enter_the_path' \
--data_split_file 'enter_the_path' --checkpoint_save_dir './checkpoints' \
--language_model 'bert' --max_people_num 6 --context_length 5 \
--batch_size 16 --learning_rate 0.000005 \
--epochs 200 --epochs_warmup 10
```
```shell
# Training example for pronoun coreference resolution
python train.py \
--task 'PCR' \
--txt_dir 'enter_the_path' --txt_labeled_dir 'enter_the_path' \
--keypoint_dir 'enter_the_path' --meta_dir 'enter_the_path' \
--data_split_file 'enter_the_path' --checkpoint_save_dir './checkpoints' \
--language_model 'bert' --max_people_num 6 --context_length 5 \
--batch_size 16 --learning_rate 0.000005 \
--epochs 200 --epochs_warmup 10
```
```shell
# Training example for mentioned player prediction
python train.py \
--task 'MPP' \
--txt_dir 'enter_the_path' --txt_labeled_dir 'enter_the_path' \
--keypoint_dir 'enter_the_path' --meta_dir 'enter_the_path' \
--data_split_file 'enter_the_path' --checkpoint_save_dir './checkpoints' \
--language_model 'bert' --max_people_num 6 --context_length 5 \
--batch_size 16 --learning_rate 0.000005 \
--epochs 200 --epochs_warmup 10
```
Descriptions of training parameters are as follows:
- `--task`: target task (STI or PCR or MPP)
- `--txt_dir`: directory of anonymized transcripts  `--txt_labeled_dir`: directory of labeled anonymized transcripts
- `--keypoint_dir`: directory of keypoints  `--meta_dir`: directory of game meta data
- `--data_split_file`: file path for data split  `--checkpoint_save_dir`: directory for saving checkpoints
- `--language_model`: language model (bert or roberta or electra)  `--max_people_num`: maximum number of players
- `--context_length`: size of conversation context  `--batch_size`: mini-batch size  `--learning_rate`: learning rate
- `--epochs`: number of total epochs  `--epochs_warmup`: number of visual warmup epochs  
- Refer to `train.py` for more details

## Testing
`test.py` evalutes the performance.

To test the model, run following command:
```shell
# Training example
python train.py \
--task 'STI' \
--txt_dir 'enter_the_path' --txt_labeled_dir 'enter_the_path' \
--keypoint_dir 'enter_the_path' --meta_dir 'enter_the_path' \
--data_split_file 'enter_the_path' --checkpoint_file 'enter_the_path' \
--language_model 'bert' --max_people_num 6 --context_length 5 \
--batch_size 16
```
Descriptions of testing parameters are as follows:
- `--task`: target task (STI or PCR or MPP)
- `--txt_dir`: directory of anonymized transcripts  `--txt_labeled_dir`: directory of labeled anonymized transcripts
- `--keypoint_dir`: directory of keypoints  `--meta_dir`: directory of game meta data
- `--data_split_file`: file path for data split  `--checkpoint_file`: file path for loading checkpoint
- `--language_model`: language model (bert or roberta or electra)  `--max_people_num`: maximum number of players
- `--context_length`: size of conversation context  `--batch_size`: mini-batch size  
- Refer to `test.py` for more details

## Pretrained Models
You can download the pretrained models.
| Dataset | Target Task | Pretrained Models |
|:------:|:------:|:------:|
| YouTube | Speaking Target Identification <br> Pronoun Coreference Resolution <br> Mentioned Player Prediction |  [Baseline-BERT](https://www.dropbox.com/scl/fi/ftaf0zsfqrv3fj96qw15v/youtube_STI_bert.pt?rlkey=cdtm5mf7xiaz9eezvc5vk6aqv&st=9cfryiy5&dl=0) / [Baseline-RoBERTa](https://www.dropbox.com/scl/fi/jixtzayoo89tmggq4ey4j/youtube_STI_roberta.pt?rlkey=0nw5erlv6qqq1dcky63nggdzo&st=mcynx60p&dl=0) / [Baseline-ELECTRA](https://www.dropbox.com/scl/fi/1gn9iezpmk2o26fqip0at/youtube_STI_electra.pt?rlkey=5gotyusir4jj5d5609iihjknu&st=m1irh7pp&dl=0) <br> [Baseline-BERT](https://www.dropbox.com/scl/fi/eon86vz6tuhav1s86l0ps/youtube_PCR_bert.pt?rlkey=lq8rjqr0dm4ulvzfkecy1rxcp&st=qjkso02z&dl=0) / [Baseline-RoBERTa](https://www.dropbox.com/scl/fi/oe77o0w6kd66po1bpk0lw/youtube_PCR_roberta.pt?rlkey=z4tr89gc50klcn1ub68fzduf9&st=f6xieyko&dl=0) / [Baseline-ELECTRA](https://www.dropbox.com/scl/fi/khubufluya5l5fvu6etfr/youtube_PCR_electra.pt?rlkey=2nx3k38vqy72fr8nxzasmzlab&st=enjmyowr&dl=0) <br> [Baseline-BERT](https://www.dropbox.com/scl/fi/4hhyqk0xep9ppy7crxofo/youtube_MPP_bert.pt?rlkey=0qvevjpvec46rowtadd6b9unv&st=o2p5dw4n&dl=0) / [Baseline-RoBERTa](https://www.dropbox.com/scl/fi/0c03xedye22zs0zhctzfl/youtube_MPP_roberta.pt?rlkey=xmuf1ma8o0vwex78wkh9mpgmc&st=292ojpkd&dl=0) / [Baseline-ELECTRA](https://www.dropbox.com/scl/fi/acu0mkhzab6sva5kx5pne/youtube_MPP_electra.pt?rlkey=z71iulcsqkcn5sp5yg22s4f9t&st=6af1a356&dl=0) |
| Ego4D | Speaking Target Identification <br> Pronoun Coreference Resolution <br> Mentioned Player Prediction |  [Baseline-BERT](https://www.dropbox.com/scl/fi/m5z47d2ul2qfw64d99axj/ego4d_STI_bert.pt?rlkey=3qcnt3qn6vqqvjbj308g7zv7q&st=lit7e8vr&dl=0) / [Baseline-RoBERTa](https://www.dropbox.com/scl/fi/b4zhhi1rpyv0mulbpb52t/ego4d_STI_roberta.pt?rlkey=v4yxjcip9e6ccper27c8jv722&st=3bjs025z&dl=0) / [Baseline-ELECTRA](https://www.dropbox.com/scl/fi/cugovcxza6opgh0r8qz08/ego4d_STI_electra.pt?rlkey=r20xl2m4rxzt4oenzriodzoid&st=vhb1t42n&dl=0) <br> [Baseline-BERT](https://www.dropbox.com/scl/fi/qz7zw36vcoyodxpz37ebt/ego4d_PCR_bert.pt?rlkey=99uwcw723h5mrqrq58t0wvbv3&st=vethstrn&dl=0) / [Baseline-RoBERTa](https://www.dropbox.com/scl/fi/z9iy01u06udnpmy1nq5ms/ego4d_PCR_roberta.pt?rlkey=9eca48nlcfanys2sf3huy00k9&st=yokuyora&dl=0) / [Baseline-ELECTRA](https://www.dropbox.com/scl/fi/ut7rkbsowtbtww7811pjx/ego4d_PCR_electra.pt?rlkey=sb3t1tzdggpj17hogcpf3xzkw&st=9ci33t26&dl=0) <br> [Baseline-BERT](https://www.dropbox.com/scl/fi/i0n8aex8e2vkwat8u042r/ego4d_MPP_bert.pt?rlkey=brli1m1b1ysr7u2glbsrtlj3b&st=em8izwtg&dl=0) / [Baseline-RoBERTa](https://www.dropbox.com/scl/fi/taxtyxv07y7nic4fpr830/ego4d_MPP_roberta.pt?rlkey=cu9ag07ilmcd1dw71oehzbwvc&st=wo3mdhg9&dl=0) / [Baseline-ELECTRA](https://www.dropbox.com/scl/fi/56bjzmlgrlf4u1v701gea/ego4d_MPP_electra.pt?rlkey=2tfkqf5ns516f7nn4lupo9z75&st=gerpduo9&dl=0) |

## Citation
If you find this work useful in your research, please cite the paper:
```
@inproceedings{lee2024modeling,
  title={Modeling Multimodal Social Interactions: New Challenges and Baselines with Densely Aligned Representations},
  author={Lee, Sangmin and Lai, Bolin and Ryan, Fiona and Boote, Bikram and Rehg, James M},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
