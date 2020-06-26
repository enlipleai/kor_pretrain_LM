<p align="center"><img src="./img/enai_logo.png"></p>

# Pre-Trained Korean Language Model
NLP 발전을 위해 한글 Corpus로 Pre-train한 Language Model을 공개합니다.

* Pre-Train Model Download
  * [Small Download (18M Params)](https://drive.google.com/open?id=13D9Fnnl0ra1qjPgtSWdp1-xIs6DfJ7Zg)
    * [Small with LMHead](https://drive.google.com/file/d/1QXwQ8dg4p7Xhr2GLN4joREYNgfL86trP/view?usp=sharing)
  * [Large-V1 Download (330M Params)](https://drive.google.com/file/d/1n0B3pK8DkkBvEpEXnjUX4a523LfPtumx/view?usp=sharing)
    * [Large-V1 with LMHead](https://drive.google.com/file/d/1uPZ0LeXsxMmzfDNiZIJxBOc1XDEpV1nr/view?usp=sharing)
  * []Large-V2 Download] (https://drive.google.com/file/d/1iS657qkFhYcwP28VOGLp6tYDO1JBUQnE/view?usp=sharing)
    * [Large-V2 with LMHead] (https://drive.google.com/file/d/1lUS4oP1Kw1iCnkRuFJOolFX5czCyg4cd/view?usp=sharing)


**V2 모델의 max_seq_length는 384입니다. V2 모델 사용 시 config의 max_position_embeddings를 384로 변경하여 사용부탁드립니다.**

Large Model의 경우 Fine-Tuning Step에서도 많은 Computational resource가
필요하기 때문에 고사양 Machine이 없을 시 Fine-Tuning이 어렵습니다. 이에
따라 Benchmark를 진행한 3가지 Task(KorQuAD1.0, KorNLI, KorSTS)에 대한
Fine-Tuning Model도 공개합니다.
* Fine-Tuning Model Download
  * V1
    * [KorQuAD1.0 (EM:85.61/F1:93.89)](https://drive.google.com/file/d/1kanzo9DkHfxjXGtjq62C-ZKpsPrmoE3l/view?usp=sharing)
    * [KorNLI (acc: 81.68)](https://drive.google.com/file/d/18QP4lpoqM46PLTBHJxGdzzVSqrLT9inC/view?usp=sharing)
    * [KorSTS (spearman: 83.9)](https://drive.google.com/file/d/1nVsSXnRrr6xJjkECe9tkUptt8ynnkiAz/view?usp=sharing)
  * V2
    * [KorQuAD1.0 (EM:65.17 F1:91.77)](https://drive.google.com/file/d/1bdC-KluGeB1SxJcSZ7ie5oIlJIvtzHbf/view?usp=sharing)
    * [KorNLI (acc: 83.21)](https://drive.google.com/file/d/1R69psC-sByY7w6sllom7WBCM6u2EC5lh/view?usp=sharing)
    * [KorSTS (spearman: 84.75)](https://drive.google.com/file/d/1HDfZHp0bfPDkBrT84AquF-7n9IePxOpS/view?usp=sharing)

## Pre-train Corpus
* Small: 한국어 Wikipedia
* V1: 한국어 Wikipedia + News (88M Sentences)
* V2: 한국어 Wikipedia + News (174M Sentences)


## Model Detail
* Masking Strategy: Dynamic
  Masking([RoBERTa](https://arxiv.org/abs/1907.11692)) + n-gram
  Masking([ALBERT](https://arxiv.org/abs/1909.11942))
* Additional Task: SOP(Sentence Order
  Prediction)([ALBERT](https://arxiv.org/abs/1909.11942))
* Optimizer:
  * Small: Adam Optimizer
  * Large: [Lamb Optimizer](https://arxiv.org/abs/1904.00962)
* Scheduler:
  * Small: LinearWarmup
  * Large:
    [PolyWarmup](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/schedulers.py)
* Mixed-Precision Opt Level "O1"
  ([Nvidia Apex](https://nvidia.github.io/apex/amp.html))
* Hyper-parameters

| Hyper-parameter       | Small Model | Large Model       |
|:----------------------|:------------|:------------------|
| Number of layers      | 12          | 24                |
| Hidden Size           | 256         | 1024              |
| FFN inner hidden size | 1024        | 4048              |
| Attention heads       | 8           | 16                |
| Attention head size   | 32          | 64                |
| Mask percent          | 15          | 15                |
| Learning Rate         | 0.0001      | 0.00125           |
| Warmup Proportion     | 0.1         | 0.0125            |
| Attention Dropout     | 0.1         | 0.1               |
| Dropout               | 0.1         | 0.1               |
| Batch Size            | 256         | 2048              |
| Train Steps           | 500k        | 125k(V1) 250k(V2) |


## Model Benchmark

|                               | KorQuAD1.0 (EM/F1) | KorNLI (acc) | KorSTS (spearman) |
|:-----------------------------:|:------------------:|:------------:|:-----------------:|
| multilingual-BERT (Base Size) |    70.42/90.25     |    76.33     |       77.90       |
|      KoBERT (Base Size)       |    52.81/80.27     |    79.00     |       79.64       |
|     KoELECTRA (Base Size)     |    61.10/89.59     |    80.85     |       83.21       |
|      HanBERT (Base Size)      |    78.74/92.02     |    80.89     |       83.33       |
|       Ours (Small Size)       |    78.98/88.20     |    74.67     |       74.53       |
|       Ours (Large Size)       |  **85.61/93.89**   |    81.68     |       83.90       |
| Ours-V2 (Large Size) 125k steps |  65.15/91.82     |    82.14     |       84.27       |
| Ours-V2 (Large Size) 250k steps |  65.17/91.77     |  **83.21**   |     **84.75**     |


**V2 모델은 형태소분석기를 사용하지 않았기때문에 KorQuAD Task에서 EM이 낮습니다.**
**Fine-tuning step의 pre-processing 또는 post-processing에 형태소분석기를 추가하여 이를 개선할 수 있습니다. KorNLI, KorSTS Task에서는 V2 모델의 성능이 향상된것을 확인할 수 있습니다.**

* **Fine-tuning Setting (Ours Model)**
  * Optimizer: Adam
  * Scheduler: LinearWarmup
  * Mixed Precision Opt Level "O2"
  * KorQuAD1.0
    * lr: 5e-5(V1) 3e-5(V2)
    * epochs: 4(V1) 2(V2)
    * batch size: 16
  * KorNLI
    * lr: 2e-5
    * epochs: 3
    * batch size: 32
  * KorSTS
    * Fine-tuning Step에서 분산은 상당히 클 수 있으므로 상대적으로
      Dataset의 크기가 작은 KorSTS Task는 Random Seed에 대해 Grid
      Search({1~10})를 사용하여 가장 성능이 좋은 Model을 사용하였습니다.
      ([Reference](https://arxiv.org/abs/2002.06305))
    * lr: 3e-5
    * epochs: 10
    * batch size: 16(V1) 32(V2, Small)
    * best random seed: 9(V1) 3(V2) 7(Small)

## Example Scripts
**KorQuAD1.0**
* Train
```shell
python3 run_qa.py \
  --checkpoint $MODEL_FILE \
  --config_file $CONFIG_FILE \
  --vocab_file $VOCAB_FILE \
  --train_file data/korquad/KorQuAD_v1.0_train.json \
  --max_seq_length 512 \
  --doc_stride 128 \
  --max_query_length 64 \
  --max_answer_length 30 \
  --per_gpu_train_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 4.0 \
  --adam_epsilon 1e-6 \
  --warmup_proportion 0.1
```
* Eval
```shell
python3 eval_qa.py \
  --checkpoint $MODEL_FILE \
  --config_file $CONFIG_FILE \
  --vocab_file $VOCAB_FILE \
  --predict_file data/korquad/KorQuAD_v1.0_dev.json \
  --max_seq_length 512 \
  --doc_stride 64 \
  --max_query_length 64 \
  --max_answer_length 30 \
  --batch_size 16 \
  --n_best_size 20
```
**KorNLI**
* Train
```shell
python3 run_classifier.py \
  --data_dir data/kornli \
  --task_name kornli \
  --config_file $CONFIG_FILE \
  --vocab_file $VOCAB_FILE \
  --checkpoint $MODEL_FILE \
  --do_eval \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --warmup_proportion 0.1
```
* Eval
```shell
python3 eval_classifier.py \
  --data_dir data/kornli \
  --task_name kornli \
  --config_file $CONFIG_FILE \
  --vocab_file $VOCAB_FILE \
  --checkpoint $MODEL_FILE \
  --max_seq_length 128 \
  --eval_batch_size 32
```

**KorSTS**
* Train
```shell
python3 run_classifier.py \
  --data_dir data/korsts \
  --task_name korsts \
  --config_file $CONFIG_FILE \
  --vocab_file $VOCAB_FILE \
  --checkpoint $MODEL_FILE \
  --do_eval
  --max_seq_length 128 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 10.0 \
  --warmup_proportion 0.1
```
* Eval
```shell
python3 eval_classifier.py \
  --data_dir data/korsts \
  --task_name korsts \
  --config_file $CONFIG_FILE \
  --vocab_file $VOCAB_FILE \
  --checkpoint $MODEL_FILE \
  --max_seq_length 128 \
  --eval_batch_size 32
```

## Acknowledgement
본 연구는 과학기술정보통신부 및 정보통신산업진흥원의 ‘고성능 컴퓨팅 지원’ 사업으로부터 지원받아 수행하였음  
Following(or This research) was results of a study on the "HPC Support" Project, supported by the ‘Ministry of Science and ICT’ and NIPA.

## Reference
* [Google BERT](https://github.com/google-research/bert)
* [Huggingface Transformers](https://github.com/huggingface/transformers)
* [Nvidia BERT](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT)
* [Nvidia Apex](https://nvidia.github.io/apex/index.html)
* [RoBERTa](https://arxiv.org/abs/1907.11692)
* [ALBERT](https://arxiv.org/abs/1909.11942)
* [KoBERT](https://github.com/SKTBrain/KoBERT)
* [KoELECTRA](https://github.com/monologg/KoELECTRA)
* [KorNLUDatasets](https://github.com/kakaobrain/KorNLUDatasets)


---
* 추가적으로 궁금하신점은 해당 repo의 issue를 등록해주시거나 ekjeon@enliple.com으로 메일 주시면 답변 드리겠습니다.
