#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/5/5 09:55
@project: LucaOneTasks
@file: evaluator
@desc: evaluator for LucaOneTasks
'''
import sys
import logging
import torch
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from utils import to_device, print_shape, process_outputs, print_batch, eval_metrics, sample_size, \
        save_prediction_results_during_training
    from multi_files_stream_dataloader import *
except ImportError:
    from src.utils import to_device, print_shape, process_outputs, print_batch, eval_metrics, sample_size, \
        save_prediction_results_during_training
    from src.multi_files_stream_dataloader import *
logger = logging.getLogger(__name__)


def evaluate(args, model, parse_row_func, batch_data_func, prefix="", log_fp=None):
    '''
    evaluation
    :param args:
    :param model:
    :param parse_row_func:
    :param batch_data_func:
    :param prefix:
    :param log_fp:
    :return:
    '''
    if hasattr(model, "module"):
        model = model.module
    save_output_dir = os.path.join(args.output_dir, prefix)
    print("\nEvaluating information dir: ", save_output_dir)
    if args.local_rank in [-1, 0] and not os.path.exists(save_output_dir):
        os.makedirs(save_output_dir)
    dev_sample_num = sample_size(args.dev_data_dir)
    dev_dataloader = MultiFilesStreamLoader(
        args.dev_data_dir,
        args.per_gpu_eval_batch_size,
        args.buffer_size,
        parse_row_func=parse_row_func,
        batch_data_func=batch_data_func,
        task_level_type=args.task_level_type,
        input_mode=args.input_mode,
        input_type=args.input_type,
        output_mode=args.output_mode,
        label_size=args.label_size,
        dataset_type="dev",
        vector_dirpath=args.vector_dirpath,
        matrix_dirpath=args.matrix_dirpath,
        inference=False,
        header=True,
        shuffle=False
    )
    # evaluate
    if log_fp is not None:
        log_fp.write("***** Running evaluation {} *****\n".format(prefix))
        logger.info("***** Running evaluation {} *****".format(prefix))
        log_fp.write("Dev Dataset Instantaneous batch size per GPU = %d\n" % args.per_gpu_eval_batch_size)
        logger.info("Dev Dataset Instantaneous batch size per GPU = %d" % args.per_gpu_eval_batch_size)
        log_fp.write("Dev Dataset Num examples = %d\n" % dev_sample_num)
        logger.info("Dev Dataset Num examples = %d" % dev_sample_num)
        log_fp.write("#" * 50 + "\n")
        logger.info("#" * 50)
        log_fp.flush()

    nb_steps = 0
    # truth
    truths = None
    # predicted prob
    preds = None
    eval_loss = 0
    model.eval()
    done_sample_num = 0
    for step, batch in enumerate(dev_dataloader):
        # eval
        with torch.no_grad():
            batch, cur_sample_num = to_device(args.device, batch)
            done_sample_num += cur_sample_num
            output = model(**batch)
            '''
            try:
                output = model(**batch)
            except Exception as e:
                with open("evaluate_exception_info_%d" % args.local_rank, "a+") as afp:
                    afp.write(str(e) + "\n")
                    afp.flush()
                with open("evaluate_exception_input_%d" % args.local_rank, "a+") as afp:
                    afp.write(str(batch) + "\n")
                    afp.flush()
                debug_path = "./debug/dev/local_rank%s/%d/" % ("_" + str(args.local_rank) if args.local_rank >= 0 else "", step)
                if not os.path.exists(debug_path):
                    os.makedirs(debug_path)
                with open(os.path.join(debug_path, "evaluate_exception_input_details.txt"), "a+") as afp:
                    print_batch(batch, key=None, debug_path=debug_path, wfp=afp, local_rank=args.local_rank)
                    afp.flush()
                continue
            '''
            cur_loss, cur_logits, cur_output = output[:3]
            cur_loss = cur_loss.item()
            eval_loss += cur_loss
            nb_steps += 1
            print("\rEval, Batch: %06d, Sample Num: %d, Cur Loss: %0.6f, Avg Loss: %0.6f" % (step + 1, done_sample_num, cur_loss, eval_loss / nb_steps),
                  end="", flush=True)

            if args.do_metrics and "labels" in batch and batch["labels"] is not None:
                truths, preds = process_outputs(
                    args.output_mode,
                    batch["labels"],
                    cur_output,
                    truths,
                    preds,
                    ignore_index=args.ignore_index,
                    keep_seq=False
                )
    avg_loss = eval_loss / nb_steps
    all_result = {
        "avg_loss": round(float(avg_loss), 6),
        "total_loss": round(float(eval_loss), 6)
    }
    save_prediction_results_during_training("dev", truths, preds, args.output_mode,  save_output_dir)
    if args.do_metrics and truths is not None and len(truths) > 0:
        dev_metrics = eval_metrics(args.output_mode, truths, preds, threshold=0.5)
        all_result.update(
            dev_metrics
        )

    with open(os.path.join(save_output_dir, "dev_metrics.txt"), "w") as writer:
        writer.write("***** Dev results {} *****\n".format(prefix))
        writer.write("Test average loss = %0.6f\n" % avg_loss)
        writer.write("Test total loss = %0.6f\n" % eval_loss)
        for key in sorted(all_result.keys()):
            writer.write("%s = %s\n" % (key, str(all_result[key])))
    return all_result
