import argparse
import json
import time
import copy
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.imgfolder import TYPE_NEW_DATA, LaplaceDataset
from memory import Memory, TYPE_MEMORABLE_PAST, TYPE_ERROR_CORRECTION
from utils import (set_lr, save_checkpoint, load_checkpoint, save_results, weight_reset,
                   set_seed, parameters_to_vector, parameters_to_vector_filter,
                   get_data_loader, get_optimizer, get_model, cross_entropy, top1_correct,
                   topk_correct, get_device_and_comm_rank, Logger,
                   DATASET_CIFAR, DATASET_TINYIMAGENET, DATASET_IMAGENET)
from tasks import Task, define_splitcifar_task_list, define_tinyimagenet_task_list, define_imagenet_ffcv_task_list


# yapf: disable
parser = argparse.ArgumentParser()
# Training settings
parser.add_argument('--n_tasks', type=int, default=5,
                    help='number of tasks')
parser.add_argument('--n_classes', type=int, default=None,
                    help='number of classes (None means all classes)')
parser.add_argument('--shuffle_classes', action='store_true',
                    help='shuffle split classes')
parser.add_argument('--use_func_regularizer', default=False, action='store_true',
                    help='if True, use functional regularization term.')
parser.add_argument('--use_ewc_weight_regularizer', default=False, action='store_true',
                    help='if True, use EWC weight regularization term.')
parser.add_argument('--use_experience_replay', default=False, action='store_true',
                    help='if True, use experience replay term (= NN error correction).')
parser.add_argument('--choose_m2_as_subset_of_m1', default=False, action='store_true',
                    help='if True, choose M2 to be a subset of M1.')
parser.add_argument('--memory_args', type=json.loads, default={},
                    help='[JSON] arguments for the Memory class')
parser.add_argument('--dataset', type=str, default=DATASET_CIFAR,
                    choices=[
                        DATASET_CIFAR, DATASET_TINYIMAGENET, DATASET_IMAGENET
                    ],
                    help='name of dataset')
parser.add_argument('--dataset_root', type=str, default='./src/data',
                    help='root of dataset')
parser.add_argument('--train_root', type=str, default=None,
                    help='root of ImageNet train')
parser.add_argument('--val_root', type=str, default=None,
                    help='root of ImageNet val')
parser.add_argument('--epochs', type=int, default=10,
                    help='maximum number of epochs to train')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (per GPU)')
parser.add_argument('--test_batch_size', type=int, default=None,
                    help='input batch size for testing (per GPU)')
parser.add_argument('--val_data_frac', type=float, default=1.0,
                    help='fraction of validation data to use')
parser.add_argument('--arch', type=str,
                    help='name of the architecture')
parser.add_argument('--arch_args', type=json.loads, default={},
                    help='[JSON] arguments for the architecture')
parser.add_argument('--optim', type=str, default='SGD',
                    help='name of the optimizer')
parser.add_argument('--optim_args', type=json.loads, default={},
                    help='[JSON] arguments for the optimizer')
parser.add_argument('--max_grad_norm', type=float, default=None,
                    help='max norm of the gradients used in torch.nn.utils.clip_grad_norm_')
parser.add_argument('--use_lr_decay_with_early_stopping', default=False, action='store_true',
                    help='if True, use the plateauing learning rate decay schedule with early stopping.')
parser.add_argument('--lr_decay_factor', type=float, default=0.1,
                    help='factor for exponentially decaying the learning rate.')
parser.add_argument('--lr_decay_patience', type=int, default=3,
                    help='patience for decaying the learning rate.')
parser.add_argument('--early_stopping_patience', type=int, default=5,
                    help='patience for early stopping.')
parser.add_argument('--lr_decay_grace_period', type=int, default=10,
                    help='grace period before decaying the learning rate.')
parser.add_argument('--batch_mode', type=str, default=None,
                    choices=[None, 'joint', 'separate'],
                    help='train in batch mode instead of continually')
parser.add_argument('--weight_init_mode', type=str, default='continual',
                    choices=['continual', 'random'],
                    help='how to initialize the weights before each task')
parser.add_argument('--lam', type=float, default=1., help='EWC regularization hyper-parameter lambda (task 1)')
parser.add_argument('--lam2', type=float, default=None, help='EWC regularization hyper-parameter lambda (task 2)')
parser.add_argument('--lam3', type=float, default=None, help='EWC regularization hyper-parameter lambda (task 3)')
parser.add_argument('--lam4', type=float, default=None, help='EWC regularization hyper-parameter lambda (task 4)')
parser.add_argument('--lam5', type=float, default=None, help='EWC regularization hyper-parameter lambda (task 5)')
parser.add_argument('--lam6', type=float, default=None, help='EWC regularization hyper-parameter lambda (task 6)')
parser.add_argument('--lam7', type=float, default=None, help='EWC regularization hyper-parameter lambda (task 7)')
parser.add_argument('--lam8', type=float, default=None, help='EWC regularization hyper-parameter lambda (task 8)')
parser.add_argument('--lam9', type=float, default=None, help='EWC regularization hyper-parameter lambda (task 9)')
parser.add_argument('--tau', type=float, default=None, help='tau trade-off parameter for task 1')
parser.add_argument('--tau2', type=float, default=None, help='tau trade-off parameter for task 2 onwards')
parser.add_argument('--scale_tau_by_task_size', default=False, help='if True, scale tau by task size')
parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='amount of label smoothing when computing the loss')
parser.add_argument('--start_ramp', type=int, default=0,
                        help='when to start interpolating resolution')
parser.add_argument('--end_ramp', type=int, default=0,
                        help='when to stop interpolating resolution')

# Options
parser.add_argument('--result_root', type=str, default='./src/results',
                    help='directory to save results')
parser.add_argument('--save_load_best_val_checkpoint', default=False, action='store_true',
                    help='if True, saves and loads checkpoints with the lowest validation loss.')
parser.add_argument('--starting_task_id', type=int, default=1,
                    help='task ID to start with.')
parser.add_argument('--end_task_id', type=int, default=None,
                    help='task ID to end with.')
parser.add_argument('--starting_checkpoint_exp_id', type=str, default='exp',
                    help='experiment ID of checkpoint to load')
parser.add_argument('--starting_checkpoint_run_id', type=str, default='batch-joint',
                    help='run ID of checkpoint to load')
parser.add_argument('--distributed_backend', type=str, default='nccl',
                    help='backend for distributed init_process_group')
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--download', default=False, action='store_true',
                    help='if True, downloads the dataset (SplitCIFAR / TinyImageNet)')
parser.add_argument('--no_cuda', action='store_true',
                    help='disables CUDA training')
parser.add_argument('--use_amp', action='store_true',
                    help='enables mixed-precision training')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed')
parser.add_argument('--shuffle_classes_seed', type=int, default=0,
                    help='random seed for the class shuffling')
parser.add_argument('--n_workers', type=int, default=8,
                    help='number of sub processes for data loading')
parser.add_argument('--config_file', type=str, default=None,
                    help='config YAML file path')
parser.add_argument('--exp_id', type=str, default='exp',
                    help='ID for experiment')
parser.add_argument('--run_id', type=str, default=None,
                    help='If None, wandb.run.id will be used.')
parser.add_argument('--master_port', type=int, default=0,
                    help='master port to use for DDP (+6000, which is the default)')
# yapf: enable


def main(args, args2, model, scaler, logger, task_list, memory, laplace_model):
    observed_tasks = []  # List of tasks seen so far
    for task in task_list:
        if args.weight_init_mode == 'random' or args.batch_mode is not None:
            # Reset model weights
            model.apply(weight_reset)
            if args2.is_master:
                sum_all_weights = sum([m.numel() for m in model.parameters()])
                print(f'\nReset model weights randomly (sum of all weights: {sum_all_weights}).\n')

        if task.id > 1:
            if args.use_func_regularizer:
                # Convert targets of current training set (with all tasks) to soft targets
                task.train_dataset.convert_to_soft_targets()
                for idx, past_task in enumerate(memory.observed_tasks):
                    # Set past model predictions as targets for memorable past
                    task.train_dataset.set_memorable_points(idx, past_task.memorable_points_indices, past_task.mean, past_task.memorable_points_types)
                    if args.dataset == DATASET_IMAGENET:
                        from data.ffcv_utils import create_train_loader
                        task.train_loader, _ = create_train_loader(
                            train_dataset=args.train_root,
                            num_workers=args.n_workers,
                            batch_size=args.batch_size,
                            in_memory=True,
                            start_ramp=args.start_ramp,
                            end_ramp=args.end_ramp,
                            gpu=args2.gpu,
                            indices=task.train_dataset.indices_.cpu())

                    else:
                        task._train_loader, _ = get_data_loader(
                            train_dataset=task.train_dataset,
                            test_dataset=task.train_dataset,
                            batch_size=args.batch_size,
                            test_batch_size=args.test_batch_size,
                            n_workers=args.n_workers,
                            world_size=args2.world_size,
                            **args2.dataset_args)

            if args.use_ewc_weight_regularizer:
                # [EWC] Decay lambda
                if task.id > 2:
                    if args2.is_master:
                        print(f'Setting lambda from {args.lam:.1f} to {args.lam_all[task.id - 3]:.1f}.')
                    args.lam = args.lam_all[task.id - 3]

                # [EWC] Define EWC dataset to be previous task points left out from the memory
                ewc_dataset = copy.deepcopy(task_list[task.id - 2].train_dataset)
                if args.use_func_regularizer:
                    last_mem_points_indices = memory.observed_tasks[-1].memorable_points_indices
                else:
                    last_mem_points_indices = []

                ewc_dataset.remove_all_but_non_last_memorable_past(last_mem_points_indices)

                if len(ewc_dataset) == 0:
                    # [EWC] Skip Laplace fitting if we used all data points as memorable past
                    if args2.is_master:
                        print('-------------------------------------------------------')
                        print(f'No data to fit EWC Laplace approximation for task {task.id} ({time.time() - args2.start_time:.2f}s).')

                else:
                    # [EWC] Fit Laplace approximation to EWC dataset
                    if args2.is_master:
                        print('-------------------------------------------------------')
                        print(f'Fitting EWC Laplace approximation for task {task.id} ({time.time() - args2.start_time:.2f}s).')
                        ewc_dataset.print_dataset_info()
                        print('-------------------------------------------------------')

                    ewc_dataset = LaplaceDataset(ewc_dataset)

                    if args.dataset == DATASET_IMAGENET:
                        from data.ffcv_utils import create_train_loader, FFCVTargetDataLoader
                        ewc_ffcv_dataloader, _ = create_train_loader(
                            train_dataset=args.train_root,
                            num_workers=args.n_workers,
                            batch_size=256,
                            in_memory=False,
                            start_ramp=args.start_ramp,
                            end_ramp=args.end_ramp,
                            gpu=args2.gpu,
                            indices=ewc_dataset.indices_.cpu(),
                            drop_last=False)
                        ewc_dataloader = FFCVTargetDataLoader(ewc_ffcv_dataloader, ewc_dataset)
                    else:
                        ewc_dataloader, _ = get_data_loader(
                            train_dataset=ewc_dataset,
                            test_dataset=ewc_dataset,
                            batch_size=args.batch_size,
                            test_batch_size=args.test_batch_size,
                            n_workers=args.n_workers,
                            world_size=args2.world_size,
                            **args2.dataset_args)

                    if args.dataset == DATASET_IMAGENET:
                        ewc_dataloader_ = ewc_dataloader
                    else:
                        ewc_dataloader_ = tqdm(ewc_dataloader)
                        setattr(ewc_dataloader_, 'dataset', ewc_dataloader.dataset)
                    laplace_model.fit(ewc_dataloader_, override=False)

        # Train or load model
        observed_tasks.append(task)
        if args2.is_master:
            args2.task_train_times.append(0.0)

        test_summary = None
        if task.id < args.starting_task_id:
            load_checkpoint(model, args, f'post_task{task.id}',\
                exp_id=args.starting_checkpoint_exp_id, run_id=args.starting_checkpoint_run_id)
        else:
            test_summary = train_on_task(args, args2, model, scaler, logger, laplace_model, task, observed_tasks)

        # Test
        test_on_observed_tasks(args, args2, model, logger, task.id, observed_tasks, verbose=True, test_summary=test_summary)

        # Update memory information for functional regularization
        if args.use_func_regularizer and task.id < args.end_task_id:
            if args2.is_master:
                print(f'\nUpdating memory information for functional regularization ({time.time() - args2.start_time:.2f}s).')

            if hasattr(task.train_dataset, 'class_ids_per_task'):
                class_ids = task.train_dataset.class_ids_per_task[-1]
            else:
                class_ids = task.class_ids
            if args.dataset == DATASET_IMAGENET:
                from data.ffcv_utils import create_train_loader
                make_mem_train_loader = lambda indices: create_train_loader(
                                    train_dataset=args.train_root,
                                    num_workers=args.n_workers,
                                    batch_size=args.batch_size,
                                    in_memory=False,
                                    start_ramp=args.start_ramp,
                                    end_ramp=args.end_ramp,
                                    gpu=args2.gpu,
                                    indices=indices,
                                    drop_last=False,
                                    shuffle=False)[0]
            else:
                make_mem_train_loader = None
            memory.update_memory_info(task.train_loader,
                                      task.train_dataset,
                                      class_ids=class_ids,
                                      is_distributed=args2.is_distributed,
                                      make_mem_train_loader=make_mem_train_loader)

        # Save checkpoints and result metrics
        if args2.is_master:
            save_checkpoint(model, args, f'post_task{task.id}', data=None)
        save_results(args, f'post_task{task.id}', args2.results)
        task.clear_train_loader()


def test_and_log(args, args2, model, logger, task, observed_tasks, epoch, last_iteration, train_acc, train_acc_top5, train_loss, val_acc, val_loss, optimizer, log_key):
    summary = test_on_observed_tasks(args, args2, model, logger, task.id, observed_tasks, verbose=False)

    if args2.is_master:
        # test
        test_accs = [summary['test'][f'task{t.id}']['acc'] for t in observed_tasks]
        test_acc_avg, test_loss = summary['test']['avg']['acc'], summary['test']['avg']['loss']

        log = {
            'epoch': epoch,
            'iteration': last_iteration,
            'time': time.time() - args2.start_time,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'test_acc (avg)': test_acc_avg,
            'test_loss': test_loss,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
        }
        log.update({f'test_acc (t={t.id})': acc for t, acc in zip(observed_tasks, test_accs)})

        if args.dataset == DATASET_IMAGENET:
            test_accs_top5 = [summary['test'][f'task{t.id}']['acc_top5'] for t in observed_tasks]
            test_acc_avg_top5 = summary['test']['avg']['acc_top5']
            log['train_acc5'] = train_acc_top5
            log['test_acc5 (avg)'] = test_acc_avg_top5
            log.update({f'test_acc5 (t={t.id})': acc for t, acc in zip(observed_tasks, test_accs_top5)})

        args2.results[f'train_for_task{task.id}'][last_iteration] = log
        logger.print_report(log)

    return summary


def train_on_task(args, args2, model, scaler, logger, laplace_model, task: Task, observed_tasks):
    start_epoch = 1
    last_iteration = 0
    last_iteration_eval = -1
    log_key = f'task_{task.id}'
    if args.use_func_regularizer:
        task.train_dataset.set_taus_and_loss_weights(args.tau, args.tau2, args2.n_task_data, args.scale_tau_by_task_size)

    if args2.is_master:
        print('-------------------------------------------------------')
        print(f'Training for task {task.id} ({time.time() - args2.start_time:.2f}s).')
        task.train_dataset.print_dataset_info()
        print('-------------------------------------------------------')

    # initialize optimizer
    if args.dataset == DATASET_IMAGENET:
        from data.ffcv_utils import create_optimizer
        optimizer = create_optimizer(model, optimizer=args.optim,
                                     momentum=args.optim_args['momentum'],
                                     weight_decay=args.optim_args['weight_decay'])
    else:
        optimizer, args.optim_args = get_optimizer(args.optim,
                                                    model.parameters(),
                                                    args.optim_args)

    if args2.is_master:
        logger.update_templates(['epoch',
                                 'iteration',
                                 'time',
                                 'train_acc'] + (
                                    ['train_acc5'] if args.dataset == DATASET_IMAGENET else []) + [
                                 'train_loss'] + [
                                 f'test_acc (t={t.id})' for t in observed_tasks] + [
                                 'test_acc (avg)'] + (
                                    [f'test_acc5 (t={t.id})' for t in observed_tasks] + [
                                    'test_acc5 (avg)'] if args.dataset == DATASET_IMAGENET else []) + [
                                 'test_loss',
                                 'val_acc',
                                 'val_loss',
                                 'learning_rate'])
        logger.print_header()
        args2.results[f'train_for_task{task.id}'] = {}

    train_loader = task.train_loader
    n_iters_per_epoch = len(train_loader)
    val_loader = getattr(task, 'val_loader', None)

    test_summary = None
    val_beat_counts = 0     # number of times val accuracy has not improved
    best_val_loss, test_loss = float('inf'), float('inf')
    val_acc, val_loss = 0, 0
    if val_loader is not None:
        val_acc, _, val_loss = test_on_dataloader(args, args2, model, val_loader, dataset=task.val_dataset)
    train_acc, train_acc_top5, train_loss = test_on_dataloader(args, args2, model, train_loader, dataset=task.train_dataset)
    stop_training = False
    best_model_state_dict = copy.deepcopy(model.state_dict())

    # test before training
    test_and_log(args, args2, model, logger, task, observed_tasks, 0, last_iteration, train_acc, train_acc_top5, train_loss, val_acc, val_loss, optimizer, log_key)

    # run training
    train_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if args.dataset == DATASET_IMAGENET:
            from data.ffcv_utils import get_resolution
            res = get_resolution(epoch=epoch - 1, start_ramp=args.start_ramp, end_ramp=args.end_ramp)
            task.decoder.output_size = (res, res)

        # train
        iteration = last_iteration + 1
        train_acc, train_acc_top5, train_loss, val_acc, val_loss, last_iteration_eval = train_on_dataloader(
                args, args2, model, scaler, laplace_model, epoch, iteration, train_loader, task.train_dataset,
                optimizer, log_key, task.id, observed_tasks, val_loader)
        last_iteration = epoch * n_iters_per_epoch

        # validate (if applicable)
        if val_loader is not None:
            if last_iteration != last_iteration_eval:
                val_acc, _, val_loss = test_on_dataloader(args, args2, model, val_loader, dataset=task.val_dataset)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                val_beat_counts = 0

                # Save model checkpoint at best val loss
                best_model_state_dict = copy.deepcopy(model.state_dict())
                if args2.is_master and args.save_load_best_val_checkpoint:
                    save_checkpoint(model, args, f'post_task{task.id}', verbose=False)
            elif epoch >= args.lr_decay_grace_period:
                val_beat_counts += 1
            
            # decay learning rate if required
            if args.use_lr_decay_with_early_stopping:
                optimizer, _, stop_training = set_lr(optimizer, count=val_beat_counts, decay_patience=args.lr_decay_patience, early_stopping_patience=args.early_stopping_patience, time=time.time() - train_time, decay_factor=args.lr_decay_factor, is_master=args2.is_master)

        # test (if applicable)
        if (last_iteration != last_iteration_eval) or (epoch == args.epochs):
            test_summary = test_and_log(args, args2, model, logger, task, observed_tasks, epoch, last_iteration, train_acc, train_acc_top5, train_loss, val_acc, val_loss, optimizer, log_key)

        stop_training = stop_training or (epoch == args.epochs)

        if np.isnan(train_loss) or np.isnan(test_loss):
            stop_training = True
            if args2.is_master:
                logger.status_infeasible()
                raise RuntimeError('Detected NaN.')

        if stop_training:
            break

    if args2.is_master:
        # store time it took to train on this task
        args2.task_train_times[-1] = time.time() - train_time

    if val_loader is not None and args.save_load_best_val_checkpoint:
        # Load model checkpoint at best val loss
        model.load_state_dict(best_model_state_dict)
        # if args2.is_master:
        #     load_checkpoint(model, args, f'post_task{task.id}')

    return test_summary


def train_on_dataloader(args, args2, model, scaler, laplace_model, epoch, iteration, train_loader, train_dataset, optimizer, log_key, task_id=None, observed_tasks=None, val_loader=None):
    model.train()

    if args2.is_distributed:
        # deterministically shuffle based on epoch
        train_loader.sampler.set_epoch(epoch)

    if args.dataset == DATASET_IMAGENET:
        from data.ffcv_utils import get_lr
        lr_start, lr_end = get_lr(epoch - 1, args.epochs, args.optim_args), get_lr(epoch, args.epochs, args.optim_args)
        iters = len(train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

    total_correct = 0
    total_correct_top5 = 0
    total_loss = 0
    total_n_examples = 0
    val_acc, val_loss = 0, 0
    last_iteration_eval = -1

    for batch_idx, batch in enumerate(train_loader):
        if args.dataset == DATASET_IMAGENET:
            inputs, data_ids = batch[0], batch[1]
            extra_data = train_dataset.get_extra_data(data_ids)
            targets, true_targets, data_types, class_ids, weight = extra_data
        else:
            inputs, targets = batch[0].to(args2.device), batch[1].to(args2.device)
            true_targets = batch[2].to(args2.device)
            data_types = batch[3].to(args2.device)
            class_ids = batch[4].to(args2.device)
            weight = None if len(batch) < 6 else batch[5].to(args2.device)

        if args.dataset == DATASET_IMAGENET:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lrs[batch_idx]

        optimizer.zero_grad(set_to_none=True)

        # [AMP] forward and backward under autocast if args.use_amp is True
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            outputs = model(inputs)

            # identify / separate different types of data points
            new_data_idx = data_types == TYPE_NEW_DATA
            memory_idx = data_types == TYPE_MEMORABLE_PAST
            error_memory_idx = data_types == TYPE_ERROR_CORRECTION
            assert sum([len(new_data_idx.nonzero(as_tuple=True)[0]),
                        len(memory_idx.nonzero(as_tuple=True)[0]),
                        len(error_memory_idx.nonzero(as_tuple=True)[0])
                        ]) == inputs.shape[0], 'Separation of different types of data points failed!'

            if task_id > 1 and args.use_func_regularizer:
                # apply temperature scaling to predictions on memorable past
                outputs[memory_idx] /= args.memory_args["temp"]

            new_data_idx = torch.logical_or(new_data_idx, memory_idx)
            if args.use_experience_replay:
                # compute functional regularization term not just over M1, but also over M2
                new_data_idx = torch.logical_or(new_data_idx, error_memory_idx)

            _weight = None if weight is None else weight[new_data_idx]
            loss = cross_entropy(outputs[new_data_idx], targets[new_data_idx], class_ids[new_data_idx], _weight, "mean", args.label_smoothing)

            if args.use_ewc_weight_regularizer:
                # [EWC] Subtract log prior of Laplace approximation from loss
                param_to_vec = parameters_to_vector_filter if args.dataset == DATASET_IMAGENET else parameters_to_vector
                mean = param_to_vec(model)
                loss -= args.lam * laplace_model.log_prob(mean) / len(train_dataset)

            if args.use_experience_replay:
                n_error_memory = len(error_memory_idx.nonzero(as_tuple=True)[0])
                if n_error_memory > 0:
                    logits_memory = outputs.gather(1, class_ids)[error_memory_idx].unsqueeze(1)
                    residual_memory = (targets - true_targets)[error_memory_idx].unsqueeze(2)
                    nn_error_correction_term = torch.bmm(logits_memory, residual_memory).flatten().dot(weight[error_memory_idx_, 0])
                    loss += nn_error_correction_term / len(targets)

        scaler.scale(loss).backward()

        if args.max_grad_norm is not None:
            # [AMP] unscale the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=args.max_grad_norm)

        # update params with the gradient of loss
        scaler.step(optimizer)

        # [AMP] update the scale for next iteration.
        scaler.update()

        # Convert soft to hard targets if applicable
        if getattr(train_dataset, "use_soft_targets", False):
            targets = train_dataset.get_hard_targets(targets)
        correct = top1_correct(outputs, targets, class_ids)
        correct_top5 = topk_correct(outputs, targets, (5,), class_ids)[0] if args.dataset == DATASET_IMAGENET else torch.tensor(0)
        n_examples = len(inputs)
        loss *= n_examples

        # [COMM] all-reduce results
        if args2.is_distributed:
            # pack
            packed_tensor = torch.tensor(
                [loss, correct, n_examples]
            ).to(args2.device)
            # all-reduce
            dist.all_reduce(packed_tensor)
            # unpack
            loss = packed_tensor[0]
            correct = packed_tensor[1]
            n_examples = int(packed_tensor[2].item())

        total_correct += correct.item()
        total_correct_top5 += correct_top5.item()
        total_loss += loss.item()
        total_n_examples += n_examples
        iteration += 1

    acc = total_correct / total_n_examples
    acc_top5 = total_correct_top5 / total_n_examples
    loss = total_loss / total_n_examples
    return acc, acc_top5, loss, val_acc, val_loss, last_iteration_eval


def test_on_observed_tasks(args, args2, model, logger, last_task_id, observed_tasks, verbose=True, test_summary=None):
    if args2.is_master and verbose:
        print(f'\nTesting on all {len(observed_tasks)} tasks observed so far ({time.time() - args2.start_time:.2f}s).')
        templates = ['task_id', 'acc', 'loss', 'time']
        if args.dataset == DATASET_IMAGENET:
            templates.insert(2, 'acc_top5')
        logger.update_templates(templates)
        logger.print_header()

    n_observed_tasks = len(observed_tasks)
    total_acc = total_acc_top5 = total_loss = 0
    log_summary = {}
    all_accs = []
    for i, task in enumerate(observed_tasks):
        if test_summary is None:
            acc, acc_top5, loss = test_on_dataloader(args, args2, model, task.test_loader, task.test_dataset)
        else:
            test_summary_task = test_summary['test'][f'task{task.id}']
            acc, acc_top5, loss = test_summary_task['acc'], test_summary_task.get('acc_top5', 0.0), test_summary_task['loss']
        total_acc += acc
        total_acc_top5 += acc_top5
        total_loss += loss
        if args2.is_master:
            log = {
                'acc': acc,
                'loss': loss,
                'time': args2.task_train_times[i],
                'task_id': task.id,
            }
            if args.dataset == DATASET_IMAGENET:
                log['acc_top5'] = acc_top5
            log_summary[f'task{task.id}'] = log
            if verbose:
                logger.print_report(log)
            all_accs.append(acc)

    if args2.is_master:
        log = {
            'acc': total_acc / n_observed_tasks,
            'loss': total_loss / n_observed_tasks,
            'time': np.mean(args2.task_train_times),
            'task_id': 'avg',
        }
        if args.dataset == DATASET_IMAGENET:
            log['acc_top5'] = total_acc_top5 / n_observed_tasks
        log_summary['avg'] = log
        args2.results[f'test_after_task{last_task_id}'] = log_summary
        if verbose:
            logger.print_report(log)

    return {'test': log_summary}


def test_on_dataloader(args, args2, model, test_loader, dataset):
    model.eval()

    test_loss = 0
    correct = 0
    correct_top5 = 0
    with torch.no_grad():
        for batch in test_loader:
            if args.dataset == DATASET_IMAGENET:
                inputs, data_ids = batch[0], batch[1]
                targets = dataset.get_targets(data_ids)
                task_ids = dataset.task_ids[dataset._idx(data_ids)]
                class_ids = dataset.class_ids_per_task[task_ids]
            else:
                inputs, targets = batch[0].to(args2.device), batch[1].to(args2.device)
                class_ids = batch[4].to(args2.device)

            # Convert soft to hard targets if applicable
            if getattr(dataset, "use_soft_targets", False):
                targets = dataset.get_hard_targets(targets)

            # [AMP] forward under autocast if args.use_amp is True
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                outputs = model(inputs)
                # use test-time augmentation over left/right flips
                outputs += model(torch.flip(inputs, dims=[3]))
                test_loss += cross_entropy(outputs, targets, class_ids, label_smoothing=args.label_smoothing)
                correct += top1_correct(outputs, targets, class_ids)
                correct_top5 += topk_correct(outputs, targets, (5,), class_ids)[0] if args.dataset == DATASET_IMAGENET else torch.tensor(0)

    test_loss /= len(test_loader)
    if args2.is_distributed:
        # pack
        packed_tensor = torch.tensor([test_loss, correct]).to(args2.device)
        # all-reduce
        dist.all_reduce(packed_tensor)
        # unpack
        test_loss = (packed_tensor[0] / args2.world_size)
        correct = packed_tensor[1]

    test_size = len(test_loader.next_traversal_order()) if args.dataset == DATASET_IMAGENET else len(test_loader.dataset)
    test_acc = correct.item() / test_size
    test_acc_top5 = correct_top5.item() / test_size

    return test_acc, test_acc_top5, test_loss.item()


def run_experiment(args):
    args2 = argparse.Namespace()
    args2.start_time = time.time()
    args2.task_train_times = []
    args2.results = {}

    # check argument validity
    if args.use_func_regularizer and args.tau == 0:
        args.use_func_regularizer = False
    if args.use_func_regularizer:
        assert args.batch_mode is None, 'Batch mode cannot be used with functional regularization.'
        assert args.memory_args['n_memorable_points'] is not None, 'n_memorable_points must be set.'

    # [COMM] get MPI rank
    args2.device, args2.rank, args2.world_size = get_device_and_comm_rank(args.no_cuda,
                                                           args.distributed_backend,
                                                           args.local_rank,
														   args.run_id,
                                                           args.master_port)
    args2.is_distributed = args2.world_size > 1
    args2.is_master = args2.rank == 0
    args2.gpu = 0

    # set some other parameters
    args.data_frac = {
        'train': 1.0,
        'test': 1.0,
        'val': args.val_data_frac,
    }
    args.n_data_per_class = {'train': 400, 'test': 50, 'val': 100}

    if args.seed is not None:
        set_seed(args.seed)

    # initialize model
    model, args.arch_args = get_model(args.arch, args.arch_args, args.dataset)

    if args.dataset == DATASET_IMAGENET:
        from data.ffcv_utils import BlurPoolConv2d, start_end_ramps
        def apply_blurpool(mod: torch.nn.Module):
            for (name, child) in mod.named_children():
                if isinstance(child, torch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                    setattr(mod, name, BlurPoolConv2d(child))
                else: apply_blurpool(child)
        apply_blurpool(model)
        model = model.to(memory_format=torch.channels_last)
        args.start_ramp, args.end_ramp = start_end_ramps[args.epochs]

    model = model.to(args2.device)
    if args2.is_distributed:
        model = DDP(model, device_ids=[args2.device])

    if args.use_func_regularizer:
        # initialize memory
        args.memory_args['use_experience_replay'] = args.use_experience_replay
        args.memory_args['choose_m2_as_subset_of_m1'] = args.choose_m2_as_subset_of_m1
        memory = Memory(model, **args.memory_args)
    else:
        memory = None

    # [AMP] initialize gradient scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # setup dataset args
    args2.dataset_args = dict(
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        download=args.download)

    # setup task args
    task_args = dict(
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        n_workers=args.n_workers,
        world_size=args2.world_size,
        dataset_args=args2.dataset_args)

    # define sequence of tasks
    if args.dataset == DATASET_CIFAR:
        define_task_list_fun = define_splitcifar_task_list
    elif args.dataset == DATASET_TINYIMAGENET:
        define_task_list_fun = define_tinyimagenet_task_list
    elif args.dataset == DATASET_IMAGENET:
        define_task_list_fun = define_imagenet_ffcv_task_list
    task_list = define_task_list_fun(args, args2.dataset_args, task_args, is_master=args2.is_master)
    task_list = task_list[:args.end_task_id]

    # save initial number of data points in each task as a reference for scaling tau
    args2.n_task_data = []
    for task in task_list:
        data = task.train_dataset
        args2.n_task_data.append(data.get_n_task_data() if hasattr(data, 'get_n_task_data') else len(data))

    if args.use_ewc_weight_regularizer:
        from laplace import Laplace
        from laplace.curvature import AsdlEF

        # [EWC] initialize Laplace approximation
        param_to_vec = parameters_to_vector_filter if args.dataset == DATASET_IMAGENET else parameters_to_vector
        prior_mean = torch.zeros_like(param_to_vec(model))
        model_ = model.module if isinstance(model, DDP) else model
        laplace_model = Laplace(
            model_, 'classification',
            subset_of_weights='all',
            hessian_structure='diag',
            prior_mean=prior_mean,
            prior_precision=1e-3,
            backend=AsdlEF
        )

        # [EWC] define sequence of lambda trade-off parameters
        args.lam_all = [args.lam2, args.lam3, args.lam4, args.lam5, args.lam6, args.lam7, args.lam8, args.lam9]
        lam = args.lam
        for i, _lam in enumerate(args.lam_all):
            if _lam is None:
                args.lam_all[i] = lam
            else:
                lam = _lam
    else:
        laplace_model = None

    # clean up config to record
    unnecessary_args_keys = [
        'dataset_root', 'no_cuda', 'download'
    ]
    if not args2.is_distributed:
        unnecessary_args_keys.extend(['distributed_backend'])
    if not args.use_func_regularizer:
        unnecessary_args_keys.append('memory_args')

    for k in unnecessary_args_keys:
        delattr(args, k)

    if args2.is_master:
        # initialize logger
        print('---------------------------')
        logger = Logger()
        assert args.run_id is not None, 'run_id is not specified.'

        # all config
        print('===========================')
        if args2.is_distributed:
            print('Distributed training')
            print(f'world_size: {dist.get_world_size()}')
            print(f'backend: {dist.get_backend()}')
            print('---------------------------')
        for key, val in vars(args).items():
            if key == 'n_tasks':
                continue
            else:
                print(f'{key}: {val}')
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
        print(f'device: {device_name}')
        print('---------------------------')
        print(f'n_tasks: {args.n_tasks}')
        for _task in task_list:
            if _task.class_ids == list(range(_task.class_ids[0], _task.class_ids[-1]+1)):
                train_classes_str = f'classes: {_task.class_ids[0]+1:3d}-{_task.class_ids[-1]+1:3d}'
            else:
                train_classes_str = f'classes: {[id for id in _task.class_ids]}'

            n_val_data = len(_task.val_loader.dataset) if getattr(_task, 'val_loader', None) is not None else 0
            n_train_data = len(_task.train_loader.next_traversal_order()) if args.dataset == DATASET_IMAGENET else len(_task.train_loader.dataset)
            n_test_data = len(_task.test_loader.next_traversal_order()) if args.dataset == DATASET_IMAGENET else len(_task.test_loader.dataset)
            if _task.class_ids == _task.class_ids_test:
                print(f'Task {_task.id:2d}: {train_classes_str} ({n_train_data}; val: {n_val_data}; test: {n_test_data}).')
            else:
                if _task.class_ids_test == list(range(_task.class_ids_test[0], _task.class_ids_test[-1]+1)):
                    val_test_classes_str = f'val/test classes: {_task.class_ids_test[0]+1:3d}-{_task.class_ids_test[-1]+1:3d}'
                else:
                    val_test_classes_str = f'val/test classes: {[id for id in _task.class_ids_test]}'
                print(f'Task {_task.id:2d}: train {train_classes_str} ({n_train_data}); {val_test_classes_str} (val: {n_val_data}; test: {n_test_data}).')
        print('===========================')
    else:
        logger = None

    args2.checkpoint_summary = {}

    if args2.is_distributed:
        # Waiting until initialization has finished
        torch.distributed.barrier()

    main(args, args2, model, scaler, logger, task_list, memory, laplace_model)

    if args2.is_master:
        print(f'=========== DONE ({time.time() - args2.start_time:.2f}s) ===========')
        logger.status_complete()


if __name__ == '__main__':
    args = parser.parse_args()

    # load config file (YAML)
    if args.config_file is not None:
        dict_args = vars(args)
        with open(args.config_file) as f:
            config = yaml.full_load(f)
        dict_args.update(config)

    print('Run experiment...')
    run_experiment(args)
