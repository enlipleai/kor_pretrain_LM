# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import json
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm

from models.modeling_bert import Config, QuestionAnswering
from utils.korquad_utils import (read_squad_examples, convert_examples_to_features, RawResult, write_predictions)
from utils.tokenization import BertTokenizer
from debug.evaluate_korquad import evaluate as korquad_eval
# The follwing import is the official SQuAD evaluation script (2.0).
# You can remove it from the dependencies if you are using this script outside of the library
# We've added it here for automated tests (see examples/test_examples.py file)


logger = logging.getLogger(__name__)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def evaluate(args, model, eval_examples, eval_features):
    """ Eval """
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    logger.info("***** Evaluating *****")
    logger.info("  Num features = %d", len(dataset))
    logger.info("  Batch size = %d", args.batch_size)

    model.eval()
    all_results = []
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    logger.info("Start evaluating!")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(dataloader, desc="Evaluating"):
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        segment_ids = segment_ids.to(args.device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
    write_predictions(eval_examples, eval_features, all_results,
                      args.n_best_size, args.max_answer_length,
                      False, output_prediction_file, output_nbest_file,
                      None, False, False, 0.0)

    expected_version = 'KorQuAD_v1.0'
    with open(args.predict_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        read_version = "_".join(dataset_json['version'].split("_")[:-1])
        if (read_version != expected_version):
            logger.info('Evaluation expects ' + expected_version +
                        ', but got dataset with ' + read_version,
                        file=sys.stderr)
        dataset = dataset_json['data']
    with open(os.path.join(args.output_dir, "predictions.json")) as prediction_file:
        predictions = json.load(prediction_file)
    logger.info(json.dumps(korquad_eval(dataset, predictions)))


def load_and_cache_examples(args, tokenizer):
    # Load data features from cache or dataset file
    examples = read_squad_examples(input_file=args.predict_file,
                                   is_training=False,
                                   version_2_with_negative=False)
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            max_seq_length=args.max_seq_length,
                                            doc_stride=args.doc_stride,
                                            max_query_length=args.max_query_length,
                                            is_training=False)
    return examples, features


def main():
    parser = argparse.ArgumentParser()
    # Parameters
    parser.add_argument("--output_dir", default='debug', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--checkpoint", default='output/korquad_3.bin', type=str,
                        help="fine-tuned model checkpoint")
    parser.add_argument("--config_file", default='data/large_config.json', type=str,
                        help="model configuration file")
    parser.add_argument("--vocab_file", default='data/large_v1_32k_vocab.txt', type=str,
                        help="tokenizer vocab file")

    parser.add_argument("--predict_file", default='data/korquad/KorQuAD_v1.0_dev.json', type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=64, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")

    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions."
                             "json output file.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.info("device: %s, n_gpu: %s, 16-bits training: %s", device, args.n_gpu, args.fp16)

    # Set seed
    set_seed(args)

    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_basic_tokenize=True, max_len=args.max_seq_length)
    config = Config.from_json_file(args.config_file)
    model = QuestionAnswering(config)
    model.load_state_dict(torch.load(args.checkpoint))
    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)
    if args.fp16:
        model.half()
    model.to(args.device)
    logger.info("Evaluation parameters %s", args)

    # Evaluate
    examples, features = load_and_cache_examples(args, tokenizer)
    evaluate(args, model, examples, features)


if __name__ == "__main__":
    main()
