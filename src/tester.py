#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/5/5 09:55
@project: LucaOneTasks
@file: tester
@desc: tester for LucaOneTasks
'''
import sys
import torch
import logging
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from utils import to_device, print_shape, process_outputs, print_batch, eval_metrics, sample_size
    from multi_files_stream_dataloader import *
except ImportError:
    from src.utils import to_device, print_shape, process_outputs, print_batch, eval_metrics, sample_size
    from src.multi_files_stream_dataloader import *
logger = logging.getLogger(__name__)


def test(args, model, parse_row_func, batch_data_func, prefix="", log_fp=None):
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
    save_output_dir = os.path.join(args.output_dir, prefix)
    print("\nTesting information dir: ", save_output_dir)
    if not os.path.exists(save_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(save_output_dir)
    if "#" in args.test_data_dir:
        test_data_dir_list = args.test_data_dir.split("#")
        test_data_dir_list = [v for v in test_data_dir_list if len(v) > 0]
    else:
        test_data_dir_list = [args.test_data_dir]
    test_sample_num_list = []
    test_dataloader_list = []
    for test_dirpath in test_data_dir_list:
        test_sample_num = sample_size(test_dirpath)
        test_dataloader = MultiFilesStreamLoader(test_dirpath,
                                                 args.per_gpu_eval_batch_size,
                                                 args.buffer_size,
                                                 parse_row_func=parse_row_func,
                                                 batch_data_func=batch_data_func,
                                                 task_level_type=args.task_level_type,
                                                 input_mode=args.input_mode,
                                                 input_type=args.input_type,
                                                 output_mode=args.output_mode,
                                                 label_size=args.label_size,
                                                 datatset_type="test",
                                                 vector_dirpath=args.vector_dirpath,
                                                 matrix_dirpath=args.matrix_dirpath,
                                                 inference=False,
                                                 header=True,
                                                 shuffle=False
                                                 )
        test_sample_num_list.append(test_sample_num)
        test_dataloader_list.append(test_dataloader)
    # Testing
    if log_fp is not None:
        log_fp.write("***** Running testing {} *****\n".format(prefix))
        logger.info("***** Running testing {} *****".format(prefix))
        log_fp.write("Test Dataset Instantaneous batch size per GPU = %d\n" % args.per_gpu_eval_batch_size)
        logger.info("Test Dataset Instantaneous batch size per GPU = %d" % args.per_gpu_eval_batch_size)
        log_fp.write("Test Dataset Num examples = %s\n" % str(test_sample_num))
        logger.info("Test Dataset Num examples = %s" % str(test_sample_num))
        log_fp.write("#" * 50 + "\n")
        logger.info("#" * 50)
        log_fp.flush()
    all_result = {}
    # 多个测试集
    for testset_idx, test_dataloader in enumerate(test_dataloader_list):
        nb_steps = 0
        # truth
        truths = None
        # predicted prob
        preds = None
        test_loss = 0
        model.eval()
        done_sample_num = 0
        for step, batch in enumerate(test_dataloader):
            # testing
            with torch.no_grad():
                batch, cur_sample_num = to_device(args.device, batch)
                done_sample_num += cur_sample_num
                output = model(**batch)
                '''
                try:
                    output = model(**batch)
                except Exception as e:
                    with open("test_exception_info_%d" % args.local_rank, "a+") as afp:
                        afp.write(str(e) + "\n")
                        afp.flush()
                    with open("test_exception_input_%d" % args.local_rank, "a+") as afp:
                        afp.write(str(batch) + "\n")
                        afp.flush()
                    debug_path = "./debug/test/local_rank%s/%d/" % (
                        "_" + str(args.local_rank) if args.local_rank >= 0 else "", step)
                    if not os.path.exists(debug_path):
                        os.makedirs(debug_path)
                    with open(os.path.join(debug_path, "test_exception_input_details.txt"), "a+") as afp:
                        print_batch(batch, key=None, debug_path=debug_path, wfp=afp, local_rank=args.local_rank)
                        afp.flush()
                    continue
                '''
                cur_loss, cur_logits, cur_output = output[:3]
                cur_loss = cur_loss.item()
                test_loss += cur_loss
                nb_steps += 1
                print("\rTest, Batch: %06d, Sample Num: %d, Cur Loss: %0.6f, Avg Loss: %0.6f" % (
                    step + 1, done_sample_num, cur_loss, test_loss / nb_steps), end="", flush=True)

                if args.do_metrics and "labels" in batch and batch["labels"] is not None:
                    truths, preds = process_outputs(args.output_mode,
                                                    batch["labels"],
                                                    cur_output,
                                                    truths,
                                                    preds,
                                                    ignore_index=args.ignore_index,
                                                    keep_seq=False)
        cur_avg_loss = test_loss / nb_steps
        cur_all_result = {
            "avg_loss": round(float(cur_avg_loss), 6),
            "total_loss": round(float(test_loss), 6)
        }
        if args.do_metrics and truths is not None and len(truths) > 0:
            cur_test_metrics = eval_metrics(args.output_mode, truths, preds, threshold=0.5)
            cur_all_result.update(
                cur_test_metrics
            )
        if testset_idx > 0:
            for item in cur_all_result.items():
                all_result[item[0] + "_%d" % (testset_idx + 1)] = item[1]
        else:
            all_result.update(
                cur_all_result
            )

    with open(os.path.join(save_output_dir, "test_metrics.txt"), "w") as writer:
        writer.write("***** Test results {} *****\n".format(prefix))
        for idx in range(len(test_dataloader_list)):
            if idx == 0:
                writer.write("Test average loss = %0.6f" % all_result["avg_loss"])
                writer.write("Test total loss = %0.6f" % all_result["total_loss"])
            else:
                writer.write("Test average loss = %0.6f" % all_result["avg_loss_%d" % (idx + 1)])
                writer.write("Test total loss = %0.6f" % all_result["total_loss_%d" % (idx + 1)])
        for key in sorted(all_result.keys()):
            writer.write("%s = %s\n" % (key, str(all_result[key])))
    return all_result
