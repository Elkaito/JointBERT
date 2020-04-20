import argparse

from trainer import Trainer
from utils import init_logger, load_tokenizer, read_prediction_text, MODEL_CLASSES, MODEL_PATH_MAP
from data_loader import load_and_cache_examples
import time
from datetime import timedelta


def main(args):

    init_logger()
    tokenizer = load_tokenizer(args)

    if args.pre_task:

        main_task = args.task
        pre_task = args.pre_task
        # Pretrain task on full dataset
        args.task = pre_task
        pre_train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
        pre_dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
        pre_test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

        trainer = Trainer(args, pre_train_dataset, pre_dev_dataset, pre_test_dataset)


        if args.do_train:
            #Pre train on pre_task 1
            trainer.train()

            if args.pre_task_2: # Pre train on task 2 if specified

                trainer.load_model() # load params from task 1
                pre_task_2 = args.pre_task_2
                # Pretrain on full dataset
                args.task = pre_task_2
                pre2_train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
                pre2_dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
                pre2_test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

                trainer.train_dataset = pre2_train_dataset
                trainer.dev_dataset = pre2_dev_dataset
                trainer.test_dataset = pre2_test_dataset
                trainer.train()



            # Train on main_task
            args.task = main_task
            args.data_dir = "./few-shot"
            train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
            dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
            test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

            trainer.train_dataset = train_dataset
            trainer.dev_dataset = dev_dataset
            trainer.test_dataset = test_dataset
            trainer.train()

        if args.do_eval:
            trainer.load_model()
            trainer.evaluate("test")

    else:
        train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
        dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
        test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

        trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

        if args.do_train:
            trainer.train()

        if args.do_eval:
            trainer.load_model()
            trainer.evaluate("test")

        if args.do_pred:
            trainer.load_model()
            texts = read_prediction_text(args)
            trainer.predict(texts, tokenizer)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Total run time: " + str(timedelta(seconds=elapsed_time)))

"""
def main(args):

    init_logger()
    tokenizer = load_tokenizer(args)

    if args.pre_task:

        main_task = args.task
        pre_task = args.pre_task
        # Train task on main task
        train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
        dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
        test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

        trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

        if args.do_train:
            trainer.train()
            trainer.load_model()
            ### train on pre task
            args.data_dir = "./data"
            args.task = pre_task
            pre_train_set = load_and_cache_examples(args, tokenizer, mode="train")
            pre_dev_set = load_and_cache_examples(args, tokenizer, mode="dev")
            pre_test_set = load_and_cache_examples(args, tokenizer, mode="test")
            trainer.train_dataset = pre_train_set
            trainer.dev_dataset = pre_dev_set
            trainer.test_dataset = pre_test_set
            trainer.train()

        if args.do_eval:
            trainer.load_model()
            args.task = main_task
            trainer.test_dataset = test_dataset
            args.data_dir = "./few-shot"
            trainer.evaluate("test")

    else:
        train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
        dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
        test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

        trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

        if args.do_train:
            trainer.train()

        if args.do_eval:
            trainer.load_model()
            trainer.evaluate("test")

        if args.do_pred:
            trainer.load_model()
            texts = read_prediction_text(args)
            trainer.predict(texts, tokenizer)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Total run time: " + str(timedelta(seconds=elapsed_time)))"""
if __name__ == '__main__':

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")

    parser.add_argument("--model_type", default="bert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    # For few-shot learning
    parser.add_argument("--K", default=None, type=int, help="train with K samples at most for every intent")
    parser.add_argument("--percent", default=None,  type=int, help="train with K samples at most for every intent")
    parser.add_argument("--pre_task", default=None, type=str, help="The name of task to pretrain on")
    parser.add_argument("--pre_task_2", default=None, type=str, help="The name of second task to pretrain on")

    # Training details
    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

    # For prediction
    parser.add_argument("--pred_dir", default="./preds", type=str, help="The input prediction dir")
    parser.add_argument("--pred_input_file", default="preds.txt", type=str, help="The input text file of lines for prediction")
    parser.add_argument("--pred_output_file", default="outputs.txt", type=str, help="The output file of prediction")
    parser.add_argument("--do_pred", action="store_true", help="Whether to predict the sentences")

    # CRF option
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")

    args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
