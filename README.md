<p align="center"><img src="./img/enai_logo.png"></p>

# Pre-Trained Korean Language Model
NLP 발전을 위해 한글 Corpus로 Pre-train한 Language Model을 공개합니다.

* [V1 Download](https://drive.google.com/file/d/1n0B3pK8DkkBvEpEXnjUX4a523LfPtumx/view?usp=sharing)
* V2 Download (In Progress)
## Pre-train Corpus
* V1: 한국어 Wikipedia + News (88M Sentences)
* V2: 한국어 Wikipedia + News (174M Sentences)


## Model Detail
* Masking Strategy: Dynamic
  Masking([RoBERTa](https://arxiv.org/abs/1907.11692)) + n-gram
  Masking([ALBERT](https://arxiv.org/abs/1909.11942))
* Optimizer: [Lamb Optimizer](https://arxiv.org/abs/1904.00962)
* Scheduler:
  [PolyWarmup](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/schedulers.py)
* Hyper-parameters

| Hyper-parameter       | Large Model       |
|:----------------------|:------------------|
| Number of layers      | 24                |
| Hidden Size           | 1024              |
| FFN inner hidden size | 4048              |
| Attention heads       | 16                |
| Attention head size   | 64                |
| Mask percent          | 15                |
| Learning Rate         | 0.00125           |
| Warmup Proportion     | 0.0125            |
| Attention Dropout     | 0.1               |
| Dropout               | 0.1               |
| Batch Size            | 2048              |
| Train Steps           | 125k(V1) 250k(V2) |


## Model Benchmark
* In Progress


## Example Scripts
* In Progress


## Reference
* [Google BERT](https://github.com/google-research/bert)
* [Huggingface Transformers](https://github.com/huggingface/transformers)
* [Nvidia BERT](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT)
* [RoBERTa](https://arxiv.org/abs/1907.11692)
* [ALBERT](https://arxiv.org/abs/1909.11942)


---
* 추가적으로 궁금하신점은 해당 repo의 issue를 등록해주시거나 ekjeon@enliple.com으로 메일 주시면 답변 드리겠습니다.
