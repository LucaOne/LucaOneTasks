#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2022/11/28 19:13
@project: LucaOneTasks
@file: trainer
@desc: trainer on training dataset for model building
'''
import os
import sys
import json
import logging
import torch
import time
import shutil
from utils import set_seed
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
import torch.distributed as dist
try:
    from .evaluator import evaluate
    from .tester import test
    from .utils import sample_size, to_device, get_lr, writer_info_tb, print_batch, lcm
except ImportError:
    from src.evaluator import evaluate
    from src.tester import test
    from src.utils import sample_size, to_device, get_lr, writer_info_tb, print_batch, lcm

logger = logging.getLogger(__name__)


def reduce_tensor(tensor, world_size):
    # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt


def train(args, train_dataloader, model_config, model, seq_tokenizer, parse_row_func, batch_data_func, train_sampler=None, log_fp=None):
    """
    building the model
    每隔一个epoch进行评估
    """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.tb_log_dir)
        if log_fp is None:
            log_fp = open(os.path.join(args.log_dir, "logs.txt"), "w")
    train_sample_num = sample_size(args.train_data_dir)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_batch_total_num = (train_sample_num + args.train_batch_size - 1) // args.train_batch_size
    if args.local_rank in [-1, 0]:
        print("Train dataset len: %d, batch size: %d, batch num: %d" % (train_sample_num, args.train_batch_size, train_batch_total_num))
        log_fp.write("Train dataset len: %d, batch size: %d, batch num: %d\n" % (train_sample_num, args.train_batch_size, train_batch_total_num))

    if args.logging_steps <= 0:
        args.logging_steps = (train_batch_total_num + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    if args.save_steps <= 0:
        args.save_steps = args.logging_steps

    t_total = args.num_train_epochs * (train_batch_total_num + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    if args.local_rank in [-1, 0]:
        log_fp.write("Train dataset t_total: %d, max_steps: %d\n" % (t_total, args.max_steps))
    if args.max_steps < t_total:
        args.max_steps = t_total
        # args.num_train_epochs = (args.max_steps * args.gradient_accumulation_steps + train_batch_total_num - 1) // train_batch_total_num

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "layernorm.weight", "layer_norm.weight", "layer.norm.weight"]
    no_decay_keys = [n for n, _ in model.named_parameters() if any(nd in n.lower() for nd in no_decay)]
    print("no_decay_keys: ")
    print(no_decay_keys)
    print("-"*50)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      betas=[args.beta1 if args.beta1 > 0 else 0.9, args.beta2 if args.beta2 > 0 else 0.98],
                      eps=args.adam_epsilon)
    print("Init lr: ", get_lr(optimizer))
    print("LR_update_strategy: %s" % args.lr_update_strategy)
    if args.lr_update_strategy == "step" and args.warmup_steps > 0:
        print("Use Warmup")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
    else:
        print("Use ExponentialLR")
        args.lr_update_strategy = "epoch"
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate if args.lr_decay_rate > 0 else 0.9)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    # Distributed training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        # find_unused_parameters=True
        find_unused_parameters = True
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=find_unused_parameters)
    if args.local_rank in [0, -1]:
        # Train
        log_fp.write("***** Running training *****\n")
        logger.info("***** Running training *****")
        log_fp.write("Train Dataset Num examples = %d\n" % train_sample_num)
        logger.info("Train Dataset Num examples = %d" % train_sample_num)
        log_fp.write("Train Dataset Num Epochs = %d\n" % args.num_train_epochs)
        logger.info("Train Dataset Num Epochs = %d" % args.num_train_epochs)
        log_fp.write("Logging Steps = %d\n" % args.logging_steps)
        logger.info("Logging Steps = %d" % args.logging_steps)
        log_fp.write("Saving Steps = %d\n" % args.save_steps)
        logger.info("Saving Steps = %d" % args.save_steps)
        log_fp.write("Train Dataset Instantaneous batch size per GPU = %d\n" % args.per_gpu_train_batch_size)
        logger.info("Train Dataset Instantaneous batch size per GPU = %d" % args.per_gpu_train_batch_size)
        log_fp.write("Train Dataset Total train batch size (w. parallel, distributed & accumulation) = %d\n" % (args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)))
        logger.info("Train Dataset Total train batch size (w. parallel, distributed & accumulation) = %d" % (args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)))
        log_fp.write("Train Dataset Gradient Accumulation steps = %d\n" % args.gradient_accumulation_steps)
        logger.info("Train Dataset Gradient Accumulation steps = %d" % args.gradient_accumulation_steps)
        log_fp.write("Train Dataset Total optimization steps = %d\n" % t_total)
        logger.info("Train Dataset Total optimization steps = %d" % t_total)
        log_fp.write("#" * 50 + "\n")
        log_fp.flush()

    optimizer.zero_grad()
    if args.local_rank in [0, -1]:
        global_step = 0
        best_metric_type = args.best_metric_type
        best_metric_flag = True
        if "loss" in best_metric_type: # argmin
            best_metric_value = 10000000.0
            best_metric_flag = False
        else: # argmax
            best_metric_value = 0.0
        best_metric_model_info = {}
        run_begin_time = time.time()
        total_loss, logging_loss = 0.0, 0.0
        real_epoch = 0
        total_use_time = 0
        cur_epoch_total_loss = 0.0
        if args.early_stop_epoch is not None and args.early_stop_epoch > 1:
            last_history_metrics = []

    for epoch in range(args.num_train_epochs):
        early_stop_flag = False
        if train_sampler:
            train_sampler.set_epoch(epoch)
        if args.local_rank in [0, -1]:
            cur_epoch_total_loss = 0.0
            print("\n=====Epoch: %06d=====" % (epoch + 1))

        cur_epoch_step = 0
        cur_epoch_loss = 0.0
        cur_epoch_time = 0.0
        done_sample_num = 0
        for step, batch in enumerate(train_dataloader):
            model.train()
            if args.local_rank in [-1, 0]:
                begin_time = time.time()
            batch, cur_sample_num = to_device(args.device, batch)
            # print(batch["input_ids"].tolist())
            # print(batch["seq_attention_masks"].tolist())
            if args.local_rank in [-1, 0]:
                done_sample_num += cur_sample_num
            # print_batch(batch)
            # print("----" * 10)
            loss, logits, output = model(**batch)
            if torch.isnan(loss).any():
                print(batch)
                print(loss)
                print(1/0)
            if args.gradient_accumulation_steps > 1:
                # The loss of each batch will be divided by gradient_accumulation_steps
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if args.n_gpu > 1:
                reduced_loss = reduce_tensor(loss.data, dist.get_world_size())
            else:
                reduced_loss = loss

            if args.local_rank in [0, -1]:
                cur_loss = reduced_loss.item()
                end_time = time.time()
                cur_use_time = end_time - begin_time
                total_use_time += cur_use_time
                total_loss += cur_loss
                logging_loss += cur_loss
                cur_epoch_total_loss += cur_loss
                cur_epoch_loss += cur_loss
                cur_epoch_time += cur_use_time
                global_step += 1
                cur_epoch_step += 1
                if global_step % args.gradient_accumulation_steps == 0:
                    print("\rTraining, Epoch: %04d, Batch: %06d, S-N: %d, C-L: %.08f, C-A-L: %.08f, G-A-L: %.08f" % (epoch + 1,
                                                                                                                     cur_epoch_step,
                                                                                                                     done_sample_num,
                                                                                                                     cur_loss,
                                                                                                                     cur_epoch_total_loss/cur_epoch_step,
                                                                                                                     total_loss/global_step), end="", flush=True)
                    if global_step % args.logging_steps == 0:
                        log_fp.write("Training, Epoch: %04d, Batch: %06d, Sample Num: %d, Cur Loss: %.08f, Cur Avg Loss: %.08f, Log Avg loss: %.08f, Global Avg Loss: %.08f, Time: %0.4f\n"
                                     % (
                                         epoch + 1,
                                         cur_epoch_step,
                                         done_sample_num,
                                         cur_loss,
                                         cur_epoch_total_loss/cur_epoch_step,
                                         logging_loss/lcm(args.logging_steps, args.gradient_accumulation_steps),
                                         total_loss/global_step,
                                         cur_use_time)
                                     )
                        log_fp.flush()
                        writer_info_tb(tb_writer,
                                       {
                                           "epoch": epoch + 1,
                                           "cur_epoch_step": cur_epoch_step,
                                           "cur_epoch_done_sample_num": done_sample_num,
                                           "cur_batch_loss": cur_loss,
                                           "global_avg_loss": total_loss/global_step,
                                           "cur_use_time": cur_use_time,
                                           "global_step": global_step,
                                           "log_avg_loss": logging_loss/lcm(args.logging_steps, args.gradient_accumulation_steps),
                                       }, global_step, prefix="logging")
                        logging_loss = 0.0

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if scheduler and args.lr_update_strategy == "step":
                    # Update learning rate schedule
                    scheduler.step()
                    if args.local_rank in [0, -1] and global_step % args.logging_steps == 0:
                        updated_lr = get_lr(optimizer)
                        print("\nCur steps: %d,  lr: %f" % (global_step, updated_lr))
                        log_fp.write("Steps: %d, Updated lr: %f\n" % (global_step, updated_lr))
                        log_fp.flush()
                        writer_info_tb(tb_writer, {"updated_lr": updated_lr}, global_step, prefix="logging")
                optimizer.zero_grad()
                # print("lr: ", get_lr(optimizer))
            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, "checkpoint-step{}".format(global_step))
                save_check_point(args, model, model_config, seq_tokenizer, output_dir)
        optimizer.step()
        optimizer.zero_grad()
        if args.lr_update_strategy == "epoch":
            scheduler.step()
            if args.local_rank in [-1, 0]:
                updated_lr = scheduler.get_last_lr()[0]
                writer_info_tb(tb_writer, {"updated_lr": updated_lr}, global_step, prefix="logging")
        if args.n_gpu > 1:
            dist.barrier()
        if args.local_rank in [-1, 0]:
            logs = {}
            update_flag = False
            # Only evaluate at local_rank=0 or single GPU
            if args.local_rank in [-1, 0] and args.evaluate_during_training and args.dev_data_dir:
                eval_result = evaluate(args, model, parse_row_func, batch_data_func,
                                       prefix="checkpoint-{}".format(global_step),
                                       log_fp=log_fp)
                print("Eval result:")
                print(eval_result)
                for key, value in eval_result.items():
                    eval_key = "eval_{}".format(key)
                    logs[eval_key] = value
                    if key == best_metric_type:
                        if best_metric_flag and best_metric_value < value or \
                                not best_metric_flag and best_metric_value > value:
                            best_metric_value = value
                            update_flag = True
                        # judge early stop
                        if args.early_stop_epoch is not None and args.early_stop_epoch > 1:
                            last_history_metrics.append(value)
                            if len(last_history_metrics) == args.early_stop_epoch:
                                if best_metric_flag:
                                    max_idx = last_history_metrics.index(max(last_history_metrics))
                                    if max_idx == 0:
                                        early_stop_flag = True
                                    else:
                                        last_history_metrics = last_history_metrics[max_idx:]
                                else:
                                    min_idx = last_history_metrics.index(min(last_history_metrics))
                                    if min_idx == 0:
                                        early_stop_flag = True
                                    else:
                                        last_history_metrics = last_history_metrics[min_idx:]

                logs["update_flag"] = update_flag
                if args.test_data_dir:
                    if update_flag:
                        best_metric_model_info.update(
                            {
                                "epoch": epoch + 1,
                                "global_step": global_step
                            }
                        )
                    test_result = test(args, model, parse_row_func, batch_data_func,
                                       prefix="checkpoint-{}".format(global_step),
                                       log_fp=log_fp)
                    print("Test result:")
                    print(test_result)
                    for key, value in test_result.items():
                        eval_key = "test_{}".format(key)
                        logs[eval_key] = value
                    if update_flag:
                        best_metric_model_info.update(logs)
            avg_batch_time = round(cur_epoch_time / cur_epoch_step, 2)
            log_fp.write("Epoch Time: %f, Avg time per batch (s): %f\n" % (cur_epoch_time, avg_batch_time))
            if scheduler is not None:
                logs["lr"] = scheduler.get_last_lr()[0]
            else:
                logs["lr"] = get_lr(optimizer)
            logs["cur_epoch_step"] = cur_epoch_step
            logs["train_global_avg_loss"] = total_loss / global_step
            logs["train_cur_epoch_loss"] = cur_epoch_loss
            logs["train_cur_epoch_avg_loss"] = cur_epoch_loss / cur_epoch_step
            logs["train_cur_epoch_time"] = cur_epoch_time
            logs["train_cur_epoch_avg_time"] = cur_epoch_time / cur_epoch_step
            logs["epoch"] = epoch + 1
            # print(logs)
            writer_info_tb(tb_writer, logs, global_step, prefix=None)
            log_fp.write(json.dumps({**logs, **{"step": global_step}}, ensure_ascii=False) + "\n")
            log_fp.write("#" * 50 + "\n")
            log_fp.flush()
            print("End epoch: %d" % (epoch + 1))
            # save checkpoint
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            if args.save_all:
                save_check_point(args, model, model_config, seq_tokenizer, output_dir)
            elif update_flag:
                if args.delete_old:
                    # delete the old CheckPoint
                    filename_list = os.listdir(args.output_dir)
                    for filename in filename_list:
                        if "checkpoint-" in filename and filename != "checkpoint-{}".format(global_step):
                            shutil.rmtree(os.path.join(args.output_dir, filename))
                save_check_point(args, model, model_config, seq_tokenizer, output_dir)

        if args.local_rank in [0, -1]:
            if scheduler is not None and args.lr_update_strategy == "epoch":
                cur_lr = scheduler.get_last_lr()[0]
            else:
                cur_lr = get_lr(optimizer)
            print("Epoch: %d, batch total: %d, lr: %0.10f" % (epoch + 1, cur_epoch_step, cur_lr))
            real_epoch += 1

        if args.n_gpu > 1:
            dist.barrier()

        # early stop
        if early_stop_flag:
            print("Early Stop at Epoch: %d " % (epoch + 1))
            print("last_history_metrics: %s" % str(last_history_metrics))
            break

    if args.local_rank in [0, -1]:
        run_end_time = time.time()
        tb_writer.close()
        log_fp.write("#" * 25 + "Best Metric" + "#" * 25 + "\n")
        log_fp.write(json.dumps(best_metric_model_info, ensure_ascii=False) + "\n")
        log_fp.write("#" * 50 + "\n")
        avg_time_per_epoch = round((run_end_time - run_begin_time)/real_epoch, 2)
        log_fp.write("Total Time: %f, Avg time per epoch(%d epochs): %f\n" % (run_end_time - run_begin_time, real_epoch, avg_time_per_epoch))
        log_fp.flush()

    if args.n_gpu > 1:
        cleanup()

    if args.local_rank in [0, -1]:
        return global_step, total_loss / global_step, best_metric_model_info

    return None, None, None


def cleanup():
    dist.destroy_process_group()
    

def save_check_point(args, model, model_config, seq_tokenizer, output_dir):
    '''
    save checkpoint
    :param args:
    :param model:
    :param seq_tokenizer
    :param model_config
    :param output_dir:
    :return:
    '''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    try:
        model_to_save.save_pretrained(output_dir)
    except Exception as e:
        '''
        model = Model()
        torch.save(model.state_dict(),path)
        state_dict = torch.load(state_dict_path)
        model = model.load_state_dict(state_dict)
        '''
        model_config.save_pretrained(output_dir)
        torch.save(model_to_save, os.path.join(output_dir, "pytorch.pt"))
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch.pth"))
    # torch.save(model_to_save, os.path.join(output_dir + "model.pth"))
    if seq_tokenizer:
        tokenizer_dir = os.path.join(output_dir, "tokenizer")
        if not os.path.exists(tokenizer_dir):
            os.makedirs(tokenizer_dir)
        seq_tokenizer.save_pretrained(tokenizer_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    print("Saving model checkpoint to %s" % output_dir)



