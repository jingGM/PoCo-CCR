import copy
import pickle
import time
import os
from os.path import join, exists
from typing import Tuple
import subprocess

import numpy as np
import torch
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import os.path as osp
from datetime import datetime, timedelta

from src.data_loader.base_class import SimpleDataClass
from utils.config import TrainingConfig, LossNames, ModelType, DataDict, DatasetType, ScheduleMethods
from utils.functions import get_device, to_device, release_cuda
from utils.logger import SummaryBoard, configure_logger
from src.model import get_model
from src.loss import Loss, RecallCalculation
from src.data_loader.data_loader import train_data_loader, evaluation_data_loader, sanity_check_data_loader, \
    registration_collate_fn_stack_mode
from src.data_loader.recall_dataset import RecallData, RerankRecallDataset


class Trainer:
    def __init__(self, cfgs: TrainingConfig):
        """
        This class is the trainner
        Args:
            cfgs: the configuration of the training class
        """
        self.name = cfgs.name
        self.evaluation_freq = cfgs.evaluation_freq
        self.max_epoch = cfgs.max_epoch
        # self.max_iteration_per_epoch = cfgs.max_iteration_per_epoch
        # self.max_evaluation_iteration_per_epoch = cfgs.max_evaluation_iteration_per_epoch

        self.iteration = 0
        self.epoch = 0
        self.training = False
        self.best_fine_recall = 0
        self.best_coarse_recall = 0

        # set up gpus
        if cfgs.gpus.device == "cuda":
            self.device = "cuda"
        else:
            self.device = get_device(device=cfgs.gpus.device)
        if 'WORLD_SIZE' in os.environ:
            print("world size: ", int(os.environ['WORLD_SIZE']))
            self.distributed = cfgs.data_loader.distributed = int(os.environ['WORLD_SIZE']) >= 1
            log_name = self.name + "-" + str(int(os.environ['WORLD_SIZE'])) + "-" + str(
                int(os.environ['LOCAL_RANK'])) + "/" + datetime.now().strftime("%m-%d-%Y-%H-%M")
        else:
            print("world size: ", 0)
            self.distributed = cfgs.data_loader.distributed = False
            log_name = self.name + "-" + datetime.now().strftime("%m-%d-%Y-%H-%M")

        # logger
        self.output_dir = cfgs.output_dir
        self.log_steps = cfgs.logger.log_steps
        self.logger = configure_logger(verbose=cfgs.logger.verbose, tensorboard_log=cfgs.logger.log_root,
                                       tb_log_name=log_name, reset_num_timesteps=cfgs.logger.reset_num_timesteps)
        self.log_summary = SummaryBoard()  # global training information for self.logger
        self.output_file = join(self.output_dir, "output.txt")

        # model
        self.model_type = cfgs.model.model_type
        self.model = get_model(config=cfgs.model, device=self.device)
        self.snapshot = cfgs.snapshot
        if self.snapshot:
            state_dict = self.load_snapshot(self.snapshot)

        self.current_rank = 0
        self._set_model_gpus(cfgs.gpus)

        # loss, optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfgs.lr, weight_decay=cfgs.weight_decay)
        self.scheduler_type = cfgs.scheduler
        if self.scheduler_type == ScheduleMethods.step:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, cfgs.lr_decay_steps, gamma=cfgs.lr_decay)
        elif self.scheduler_type == ScheduleMethods.cosine:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, eta_min=cfgs.lr_min,
                                                                                  T_0=cfgs.lr_t0, T_mult=cfgs.lr_tm)
        else:
            raise ValueError("the current scheduler is not defined")

        if self.snapshot and not cfgs.only_model:
            self.load_learning_parameters(state_dict)

        if self.device == "cuda":
            self.loss_func =Loss(cfg=cfgs.loss, model_type=cfgs.model.model_type).cuda()
        else:
            self.loss_func =Loss(cfg=cfgs.loss, model_type=cfgs.model.model_type).to(self.device)

        # setup dataloader
        if cfgs.sanity_check:
            cfgs.data_loader.type = DatasetType.sanity_check
            training_data_config = copy.deepcopy(cfgs.data_loader)
            training_data_config.max_iteration = cfgs.max_iteration_per_epoch
            self.training_data_loader = sanity_check_data_loader(cfg=training_data_config, model_type=cfgs.model.model_type)
            if self.evaluation_freq > 0:
                evaluation_data_config = copy.deepcopy(cfgs.data_loader)
                evaluation_data_config.max_iteration = cfgs.max_evaluation_iteration_per_epoch
                self.evaluation_data_loader = sanity_check_data_loader(cfg=evaluation_data_config,
                                                                       model_type=cfgs.model.model_type)
        else:
            training_data_config = copy.deepcopy(cfgs.data_loader)
            training_data_config.max_iteration = cfgs.max_iteration_per_epoch
            self.training_data_loader = train_data_loader(cfg=training_data_config, model_type=cfgs.model.model_type)
            if self.evaluation_freq > 0:
                evaluation_data_config = copy.deepcopy(cfgs.data_loader)
                evaluation_data_config.max_iteration = cfgs.max_evaluation_iteration_per_epoch
                self.evaluation_data_loader = evaluation_data_loader(cfg=evaluation_data_config,
                                                                     model_type=cfgs.model.model_type)

        # setup negative mining dataloader
        self.mining_frequency = cfgs.mining_frequency
        self.mining_start = cfgs.mining_start
        # if self.mining_frequency > 0:
        #     # data_mining_cfg.num_pos_samples = -1
        #     if cfgs.sanity_check:
        #         cfgs.data_loader.type = DatasetType.sanity_check
        #     else:
        #         cfgs.data_loader.type = DatasetType.training
        #     self.data_mining_dataset = SimpleDataClass(cfg=cfgs.data_loader)

        # setup recall
        self.recall_frequency = cfgs.recall_frequency
        if self.recall_frequency > 0:
            if cfgs.sanity_check:
                cfgs.data_loader.type = DatasetType.sanity_check
            else:
                cfgs.data_loader.type = DatasetType.test

            recall_data_config = copy.deepcopy(cfgs.data_loader)
            recall_data_config.max_iteration = cfgs.max_recall_iteration_per_epoch
            if cfgs.model.model_type == ModelType.fine_matching:
                self.recall_dataset = RerankRecallDataset(cfg=recall_data_config)
            else:
                self.recall_dataset = RecallData(cfg=recall_data_config)
            self.recall_calculator = RecallCalculation(
                distance_type=cfgs.loss.distance_type, verbose=cfgs.logger.verbose, max_recall_num=cfgs.recall_num,
                output_dir=join(recall_data_config.root, "recall_{}_{}".format(self.current_rank, self.name)),
                model_type=cfgs.model.model_type, instance_num=cfgs.data_loader.instance_size,
                rerank_frame_num=cfgs.rerank_num)

        if self.snapshot and not cfgs.only_model and self.mining_frequency > 0:
            if self.distributed:
                file_name = "negativeming_{}".format(
                    str(int(os.environ['WORLD_SIZE'])) + "-" + str(int(os.environ['LOCAL_RANK'])))
            else:
                file_name = "negativeming"
            self.training_data_loader.dataset.update_dataset(file_name=file_name)

    def load_learning_parameters(self, state_dict):
        # Load other attributes
        if 'epoch' in state_dict:
            self.epoch = state_dict['epoch']
            self.logger.info('Epoch has been loaded: {}.'.format(self.epoch))
        if 'iteration' in state_dict:
            self.iteration = state_dict['iteration']
            self.logger.info('Iteration has been loaded: {}.'.format(self.iteration))
        if 'optimizer' in state_dict and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state_dict['optimizer'])
                self.logger.info('Optimizer has been loaded.')
            except:
                print("doesn't load optimizer")
        if 'scheduler' in state_dict and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(state_dict['scheduler'])
                self.logger.info('Scheduler has been loaded.')
            except:
                print("doesn't load scheduler")

    def load_snapshot(self, snapshot):
        """
        Load the parameters of the model and the training class
        Args:
            snapshot: the complete path to the snapshot file
        """
        self.logger.info('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=torch.device(self.device))

        # Load model
        model_dict = state_dict['state_dict']
        self.model.load_state_dict(model_dict, strict=False)

        # log missing keys and unexpected keys
        snapshot_keys = set(model_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        missing_keys = model_keys - snapshot_keys
        unexpected_keys = snapshot_keys - model_keys
        if len(missing_keys) > 0:
            message = 'Missing keys: {}'.format(missing_keys)
            self.logger.error(message)
        if len(unexpected_keys) > 0:
            message = 'Unexpected keys: {}'.format(unexpected_keys)
            self.logger.error(message)
        self.logger.info('Model has been loaded.')
        return state_dict

    def save_snapshot(self, filename):
        """
        save the snapshot of the model and other training parameters
        Args:
            filename: the output filename that is the full directory
        """
        if self.distributed:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        # save model
        state_dict = {'state_dict': model_state_dict}
        torch.save(state_dict, filename)
        self.logger.info('Model saved to "{}"'.format(filename))

        # save snapshot
        state_dict['epoch'] = self.epoch
        state_dict['iteration'] = self.iteration
        snapshot_filename = osp.join(self.output_dir, str(self.name) + 'snapshot.pth.tar')
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()
        torch.save(state_dict, snapshot_filename)
        self.logger.info('Snapshot saved to "{}"'.format(snapshot_filename))

    def cleanup(self):
        dist.destroy_process_group()

    def _set_model_gpus(self, cfg):
        # self.current_rank = 0  # global rank
        # cfg.local_rank = os.environ['LOCAL_RANK']
        if self.distributed:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            print("os world size: {}, local_rank: {}, rank: {}".format(world_size, local_rank, rank))

            # this will make all .cuda() calls work properly
            torch.cuda.set_device(cfg.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=5000))
            # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
            world_size = dist.get_world_size()
            self.current_rank = dist.get_rank()
            # self.logger.info\
            print(
                'Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                % (self.current_rank, world_size))

            # synchronizes all the threads to reach this point before moving on
            dist.barrier()
        else:
            # self.logger.info\
            print('Training with a single process on 1 GPUs.')
        assert self.current_rank >= 0, "rank is < 0"

        # if cfg.local_rank == 0:
        #     self.logger.info(
        #         f'Model created, param count:{sum([m.numel() for m in self.model.parameters()])}')

        # move model to GPU, enable channels last layout if set
        self.model.cuda()
        if cfg.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if self.distributed and cfg.sync_bn:
            assert not cfg.split_bn
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if cfg.local_rank == 0:
                self.logger.info(
                    'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                    'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

        # setup distributed training
        if self.distributed:
            if cfg.local_rank == 0:
                self.logger.info("Using native Torch DistributedDataParallel.")
            self.model = DDP(self.model, device_ids=[cfg.local_rank],
                             broadcast_buffers=not cfg.no_ddp_bb,
                             find_unused_parameters=True)
            # NOTE: EMA model does not need to be wrapped by DDP

        # # setup exponential moving average of model weights, SWA could be used here too
        # model_ema = None
        # if args.model_ema:
        #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        #     model_ema = ModelEmaV2(
        #         self.model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)

    def set_train_mode(self):
        """
        set the model to the training mode: parameters are differentiable
        """
        self.training = True
        self.model.train()
        torch.set_grad_enabled(True)

    def set_eval_mode(self):
        """
        set the model to the evaluation mode: parameters are not differentiable
        """
        self.training = False
        self.model.eval()
        torch.set_grad_enabled(False)

    def optimizer_step(self):
        """
        run one step of the optimizer
        """
        self.optimizer.step()
        self.optimizer.zero_grad()

    def step(self, data_dict) -> Tuple[dict, dict]:
        """
        one step of the model, loss function and also the metrics
        Args:
            data_dict: the input data dictionary
        Returns:
            the output from the model, the output from the loss function
        """
        # start_time = time.time()
        # data_dict = to_device(data_dict, device=self.device)
        output_dict = self.model(data_dict)
        torch.cuda.empty_cache()
        eval_dict = self.loss_func.evaluate(output_dict)
        torch.cuda.empty_cache()
        loss_dict = self.loss_func(output_dict)
        torch.cuda.empty_cache()
        loss_dict.update(eval_dict)
        return output_dict, loss_dict

    def run_epoch(self):
        """
        run training epochs
        """
        self.optimizer.zero_grad()

        last_time = time.time()
        with open(self.output_file, "a") as f:
            print("Training CUDA {} Epoch {} \n".format(self.current_rank, self.epoch), file=f)
        for iteration, data_dict in enumerate(
                tqdm(self.training_data_loader, desc="Training Epoch {}".format(self.epoch))):
            # # TODO:
            # if iteration > 0:
            #     break
            # if iteration % self.max_iteration_per_epoch == 0 and iteration != 0:
            #     break
            self.iteration += 1
            # data_time = time.time()
            # print("dt: {:.4f}".format(data_time - last_time), end=", ")

            # step_time_start = time.time()
            output_dict, result_dict = self.step(data_dict=data_dict)
            torch.cuda.empty_cache()
            # print("st: {:.4f}".format(time.time() - step_time_start), end=", ")
            # for name, param in self.model.named_parameters():
            #     print(name, param.grad)

            # optimize_time_start = time.time()
            result_dict[LossNames.loss].backward()
            self.optimizer_step()
            optimize_time = time.time()
            # print("ot: {:.4f}".format(time.time() - optimize_time_start))

            result_dict = release_cuda(result_dict)
            self.log_summary.update_from_result_dict(result_dict)
            # self.log_summary.update("data_time", data_time - last_time)
            # self.log_summary.update("step_time", step_time - step_time_start)
            # self.log_summary.update("opt_time", optimize_time - optimize_time_start)
            self.log_summary.update("training_step_time", optimize_time - last_time)
            self.log_summary.update("learning rate", self.scheduler.get_last_lr())
            if iteration % self.log_steps == 0:
                summary_dict = self.log_summary.summary()
                self.logger.record_dict(summary_dict, prefix="train/")
                self.logger.record("train/iterations", self.iteration, exclude="tensorboard")
                self.logger.record("train/learning_rate", self.scheduler.get_last_lr())
                self.logger.dump(step=self.iteration)
                self.log_summary.reset_all()
            torch.cuda.empty_cache()
            last_time = time.time()

        self.scheduler.step()
        if not self.distributed or (self.distributed and self.current_rank == 0):
            os.makedirs('{}/models'.format(self.output_dir), exist_ok=True)
            self.save_snapshot('{}/models/{}_{}.pth.tar'.format(self.output_dir, self.name, self.epoch))
        try:
            bash_command = "/opt/amazon/compute_grid_utils/output_uploader /root/CocsPR/results"
            process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            print(error)
        except:
            pass

    def update_dataset_values(self, model, file_name):
        folder_path = join(self.training_data_loader.dataset.root, file_name)
        if not exists(folder_path):
            os.makedirs(folder_path)
        frame_num = 4000
        # frame_num = 100
        frames = []
        frame_global_id = []
        global_descriptors = []
        file_index = 0
        frame_descriptor_id = 0
        for iteration in tqdm(range(self.training_data_loader.dataset.batch_length()),
                              desc="Negative mining dataset generation"):
            if len(global_descriptors) >= frame_num:
                with open(join(folder_path, "{}.pkl".format(file_index)), 'wb') as handle:
                    pickle.dump(torch.stack(global_descriptors, dim=0), handle, protocol=pickle.HIGHEST_PROTOCOL)
                file_index += 1
                frame_descriptor_id = 0
                global_descriptors = []
            data_dict = self.training_data_loader.dataset.get_batch(iteration)
            input_dict = registration_collate_fn_stack_mode([data_dict])
            input_dict = to_device(input_dict, device=self.device)
            results = model.forward_recall_features(input_data=input_dict)

            frame_id = results[DataDict.frame_id].detach().cpu().numpy()
            for idx in range(len(frame_id)):
                s_id, f_id = frame_id[idx]
                frames.append([self.training_data_loader.dataset.scenes[s_id], f_id])
                frame_global_id.append([file_index, frame_descriptor_id])
                frame_descriptor_id += 1

            assert results[DataDict.pts_coarse_embeds] is not None, \
                "it's in coarse matching mode but no coarse features"
            cfts = results[DataDict.pts_coarse_embeds].detach().cpu()
            for fts in cfts:
                global_descriptors.append(fts)

        with open(join(folder_path, "{}.pkl".format(file_index)), 'wb') as handle:
            pickle.dump(torch.stack(global_descriptors, dim=0), handle, protocol=pickle.HIGHEST_PROTOCOL)
        file_index += 1
        with open(join(folder_path, "mining.pkl"), 'wb') as handle:
            pickle.dump({"ids": frames, "file_num": file_index, "global_ids": frame_global_id}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def inference_epoch(self):
        """
        run inference of one epoch, which doesn't update the parameters of the model
        """
        summary_board = SummaryBoard()
        # Inference Losses
        with open(self.output_file, "a") as f:
            print("Evaluation CUDA {} Epoch {} \n".format(self.current_rank, self.epoch), file=f)
        if (self.evaluation_freq > 0) and (self.epoch % self.evaluation_freq == 0) and (self.epoch != 0):
        # if (self.evaluation_freq > 0) and (self.epoch % self.evaluation_freq == 0):
            self.evaluation_data_loader.dataset.reset()
            for iteration, data_dict in enumerate(tqdm(self.evaluation_data_loader,
                                                       desc="Evaluation Losses Epoch {}".format(self.epoch))):
                # if iteration % self.max_evaluation_iteration_per_epoch == 0 and iteration != 0:
                #     break
                start_time = time.time()
                output_dict, result_dict = self.step(data_dict)
                torch.cuda.synchronize()
                step_time = time.time()
                # print(step_time - start_time)
                result_dict = release_cuda(result_dict)
                summary_board.update_from_result_dict(result_dict)
                summary_board.update("step_time", step_time - start_time)
                torch.cuda.empty_cache()

        if self.distributed:
            model = self.model.module
            file_name = "negativeming_{}".format(
                str(int(os.environ['WORLD_SIZE'])) + "-" + str(int(os.environ['LOCAL_RANK'])))
        else:
            model = self.model
            file_name = "negativeming"

        # if (self.recall_frequency > 0) and (self.epoch % self.recall_frequency == 0):
        if (self.recall_frequency > 0) and (self.epoch % self.recall_frequency == 0) and (self.epoch != 0):
            self.recall_dataset.reset()
            with open(self.output_file, "a") as f:
                print("Recall CUDA {} Epoch {} \n".format(self.current_rank, self.epoch), file=f)
            coarse_recalls, fine_recalls = self.recall_calculator.evaluate_recall(model=model, device=self.device,
                                                                                  dataset=self.recall_dataset)
            recall_dict = {}
            if coarse_recalls is not None:
                recall_dict.update({"coarse_recall_1": coarse_recalls[0],
                                    "coarse_recall_2": coarse_recalls[1],
                                    "coarse_recall_3": coarse_recalls[2],})
                if coarse_recalls[0] > self.best_coarse_recall:
                    self.best_coarse_recall = coarse_recalls[0]
                    self.save_snapshot('{}/{}_best_coarse.pth.tar'.format(self.output_dir, self.name))
            if fine_recalls is not None:
                recall_dict.update({"fine_recall_1": fine_recalls[0],
                                    "fine_recall_2": fine_recalls[1],
                                    "fine_recall_3": fine_recalls[2]})
                if fine_recalls[0] > self.best_fine_recall:
                    self.best_fine_recall = fine_recalls[0]
                    self.save_snapshot('{}/{}_best_fine.pth.tar'.format(self.output_dir, self.name))
            summary_board.update_from_result_dict(recall_dict)

        if (self.mining_frequency > 0) and (self.epoch % self.mining_frequency == 0) and (self.epoch >= self.mining_start) and (self.model_type != ModelType.fine_matching):
            self.training_data_loader.dataset.reset()
            with open(self.output_file, "a") as f:
                print("Negative Mining CUDA {} Epoch {} \n".format(self.current_rank, self.epoch), file=f)
            self.update_dataset_values(model, file_name)
            # start_time = time.time()
            self.training_data_loader.dataset.update_dataset(file_name=file_name)
            # second_time = time.time()
            # self.training_data_loader.dataset.update_dataset_2(file_name=file_name)
            # print("nm: ", time.time() - second_time, second_time - start_time)

        summary_dict = summary_board.summary()
        self.logger.record_dict(summary_dict, prefix="eval/")
        self.logger.dump(step=self.epoch)

    def run(self):
        """
        run the training process
        """
        torch.autograd.set_detect_anomaly(True)
        for self.epoch in range(self.epoch, self.max_epoch, 1):
            self.set_eval_mode()
            self.inference_epoch()

            self.set_train_mode()
            if self.distributed:
                self.training_data_loader.sampler.set_epoch(self.epoch)
                if self.evaluation_freq > 0:
                    self.evaluation_data_loader.sampler.set_epoch(self.epoch)
            self.run_epoch()
        self.cleanup()
