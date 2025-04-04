import collections
import time
import types
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from omegaconf import DictConfig
from rich.pretty import pretty_repr

import robot_learning.utils.general_utils as gutl
from robot_learning.data.dataclasses import Batch
from robot_learning.trainers.base_trainer import BaseTrainer
from robot_learning.utils.logger import log
from robot_learning.utils.rollouts import run_eval_rollouts


class OfflineTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.train_step = 0

    def fix_prefix(self, prefix):
        if self.cfg.log_prefix:
            return f"{self.cfg.log_prefix}/{prefix}"
        return prefix

    def train(self):
        # first eval
        if not self.cfg.skip_first_eval and self.local_rank == 0:
            self.eval(step=0)

        # set model to train
        self.model.train()
        if hasattr(self, "action_decoder"):
            self.action_decoder.train()

        train_iter = self.train_dataloader.repeat().as_numpy_iterator()

        for self.train_step in tqdm.tqdm(
            range(self.cfg.num_updates),
            desc=f"{self.cfg.name} train batches",
            disable=self.local_rank != 0,  # Only show progress on rank 0
            total=self.cfg.num_updates,
        ):
            batch_load_time = time.time()
            batch = next(train_iter)
            # put the batch on the device
            batch = gutl.to_device(batch, self.device)
            batch = Batch.create(**batch)
            batch_load_time = time.time() - batch_load_time

            # perform a single gradient step
            update_time = time.time()

            self.optimizer.zero_grad()
            if hasattr(self, "action_decoder_optimizer"):
                self.action_decoder_optimizer.zero_grad()

            if self.cfg.accelerate.use:
                with self.accelerator.autocast():
                    metrics, total_loss = self.compute_loss(batch, train=True)

                self.accelerator.backward(total_loss)

                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad_norm
                )
                self.optimizer.step()

                if hasattr(self, "action_decoder_optimizer") and (
                    self.train_step % self.cfg.train_action_decoder_every == 0
                ):
                    self.accelerator.clip_grad_norm_(
                        self.action_decoder.parameters(), self.cfg.clip_grad_norm
                    )
                    self.action_decoder_optimizer.step()
            else:
                # Use autocast for mixed precision training
                with torch.amp.autocast("cuda"):
                    metrics, total_loss = self.compute_loss(batch, train=True)

                # Scale loss and backward pass
                self.scaler.scale(total_loss).backward()

                # Unscale gradients to prepare for gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.cfg.clip_grad_norm
                )

                self.scaler.step(self.optimizer)

                # Backward pass for action decoder
                if (
                    hasattr(self, "action_decoder_optimizer")
                    and self.train_step % self.cfg.train_action_decoder_every == 0
                ):
                    self.scaler.unscale_(self.action_decoder_optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.action_decoder.parameters(),
                        max_norm=self.cfg.clip_grad_norm,
                    )
                    self.scaler.step(self.action_decoder_optimizer)

                self.scaler.update()

            # step the scheduler at the end of everything
            self.scheduler.step()

            # step the action decoder scheduler, only if we are training the action decoder
            if (
                hasattr(self, "action_decoder_scheduler")
                and self.train_step % self.cfg.train_action_decoder_every == 0
            ):
                self.action_decoder_scheduler.step()

            metrics["time/batch_load"] = batch_load_time
            metrics["time/update"] = time.time() - update_time

            # get lr
            metrics["lr"] = self.scheduler.get_last_lr()[0]

            if hasattr(self, "action_decoder_scheduler"):
                action_decoder_lr = self.action_decoder_scheduler.get_last_lr()[0]
                metrics["action_decoder_lr"] = action_decoder_lr

            # Gather metrics from all processes if using DDP
            if self.distributed:
                gathered_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        gathered_metrics[k] = self._reduce_mean(
                            torch.tensor(v, device=self.device)
                        ).item()
                    else:
                        gathered_metrics[k] = v
                metrics = gathered_metrics

            # Only log from rank 0 or if not distributed
            if self.local_rank == 0 or not self.distributed:
                self.log_to_wandb(metrics, prefix=self.fix_prefix("train/"))

                # log stats about the model params
                param_stats = defaultdict(float)
                for name, param in self.model.named_parameters():
                    param_stats[f"{name}_mean"] = param.mean().item()
                    param_stats[f"{name}_std"] = param.std().item()

                self.log_to_wandb(param_stats, prefix=self.fix_prefix("params/"))

                # log a step counter for wandb
                self.log_to_wandb(
                    {"_update": self.train_step}, prefix=self.fix_prefix("step/")
                )

                # log to terminal
                if ((self.train_step + 1) % self.cfg.log_terminal_every) == 0:
                    log(f"step: {self.train_step}, train:")
                    print(metrics)
                    # round metrics to 5 decimal places
                    metrics = {k: round(v, 5) for k, v in metrics.items()}
                    log(f"{pretty_repr(metrics)}")

            # run evaluation for each evaluation environment
            if ((self.train_step + 1) % self.eval_every) == 0:
                self.eval(step=self.train_step + 1)

                # after eval set model back to train
                self.model.train()
                if hasattr(self, "action_decoder"):
                    self.action_decoder.train()

        # final evaluation
        self.eval(step=self.cfg.num_updates)

        if self.wandb_run is not None:
            self.wandb_run.finish()

    def _reduce_mean(self, tensor):
        """Helper for gathering tensors across processes"""
        if not self.distributed:
            return tensor
        dist.all_reduce(tensor)
        return tensor / self.world_size

    def eval(self, step: int):
        # only running eval on rank 0 process
        if self.local_rank == 0 or not self.distributed:
            log(
                "\n"
                + "=" * 80
                + "\n"
                + "                          Running evaluation for "
                + f"{self.cfg.name} | Step {step}                       "
                + "\n"
                + "=" * 80,
            )

            self.model.eval()
            if hasattr(self, "action_decoder"):
                self.action_decoder.eval()

            eval_time = time.time()
            eval_iter = self.eval_dataloader.as_numpy_iterator()

            eval_metrics = collections.defaultdict(list)
            count = 0
            for batch in tqdm.tqdm(
                eval_iter,
                desc=f"{self.cfg.name} eval batches",
                disable=self.local_rank != 0,  # Only show progress on rank 0
            ):
                if self.cfg.debug and count > 10:
                    break
                count += 1
                # put the batch on the device
                batch = gutl.to_device(batch, self.device)
                batch = Batch(**batch)

                with torch.no_grad():
                    metrics, total_eval_loss = self.compute_loss(batch, train=False)

                for k, v in metrics.items():
                    eval_metrics[k].append(v)

            # average metrics over all eval batches
            for k, v in eval_metrics.items():
                eval_metrics[k] = np.mean(np.array(v))

            eval_metrics["time"] = time.time() - eval_time

            self.log_to_wandb(eval_metrics, prefix=self.fix_prefix("eval/"))

            # write evaluation metrics to log file
            with open(self.log_dir / "eval.txt", "a+") as f:
                f.write(f"{step}, {eval_metrics}\n")

            eval_metrics = {k: round(v, 5) for k, v in eval_metrics.items()}
            log(f"eval: {pretty_repr(eval_metrics)}")

            # run evaluation rollouts
            if self.cfg.run_eval_rollouts:
                rollout_metrics, *_ = run_eval_rollouts(
                    cfg=self.cfg, model=self.model, wandb_run=self.wandb_run
                )
                self.log_to_wandb(
                    rollout_metrics, prefix=self.fix_prefix("eval_rollout/")
                )

                with open(self.log_dir / "eval.txt", "a+") as f:
                    f.write(f"{step}, {rollout_metrics}\n")

                rollout_metrics = {k: round(v, 5) for k, v in rollout_metrics.items()}
                log(f"eval rollout: {pretty_repr(rollout_metrics)}")

            # also save model here
            self.save_model(ckpt_dict=self.save_dict, metrics=eval_metrics, iter=step)

            return eval_metrics
