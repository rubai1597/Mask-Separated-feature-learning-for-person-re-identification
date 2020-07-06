import os
import sys
import time
from datetime import datetime

from apex import amp

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from data import make_data_loader
from losses import make_loss
from modeling import build_model
from solver import make_optimizer, WarmupMultiStepLR, ReIDEvaluator
from utils.common import AverageMeter
from utils.config import argument_parsing
from utils.logging import Logger


def train(args):
    if args.batch_size % args.num_instance != 0:
        new_batch_size = (args.batch_size // args.num_instance) * args.num_instance
        print(f"given batch size is {args.batch_size} and num_instances is {args.num_instance}." +
              f"Batch size must be divided into {args.num_instance}. Batch size will be replaced into {new_batch_size}")
        args.batch_size = new_batch_size

    # prepare dataset
    train_loader, val_loader, num_query, train_data_len, num_classes = make_data_loader(args)

    model = build_model(args, num_classes)
    print("model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    loss_fn, center_criterion = make_loss(args, num_classes)
    optimizer, optimizer_center = make_optimizer(args, model, center_criterion)

    if args.cuda:
        model = model.cuda()
        if args.amp:
            if args.center_loss:
                model, [optimizer, optimizer_center] = \
                    amp.initialize(model, [optimizer, optimizer_center], opt_level="O1")
            else:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        if args.center_loss:
            center_criterion = center_criterion.cuda()
            for state in optimizer_center.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

    model_state_dict = model.state_dict()
    optim_state_dict = optimizer.state_dict()
    if args.center_loss:
        optim_center_state_dict = optimizer_center.state_dict()
        center_state_dict = center_criterion.state_dict()

    reid_evaluator = ReIDEvaluator(args, model, num_query)

    start_epoch = 0
    global_step = 0
    if args.pretrain != '':  # load pre-trained model
        weights = torch.load(args.pretrain)
        model_state_dict = weights["state_dict"]

        model.load_state_dict(model_state_dict)
        if args.center_loss:
            center_criterion.load_state_dict(
                torch.load(args.pretrain.replace('model', 'center_param'))["state_dict"])

        if args.resume:
            start_epoch = weights["epoch"]
            global_step = weights["global_step"]

            optimizer.load_state_dict(torch.load(args.pretrain.replace('model', 'optimizer'))["state_dict"])
            if args.center_loss:
                optimizer_center.load_state_dict(
                    torch.load(args.pretrain.replace('model', 'optimizer_center'))["state_dict"])
        print(f'Start epoch: {start_epoch}, Start step: {global_step}')

    scheduler = WarmupMultiStepLR(optimizer, args.steps, args.gamma,
                                  args.warmup_factor, args.warmup_step, "linear",
                                  start_epoch)

    current_epoch = start_epoch
    best_epoch = 0
    best_rank1 = 0
    best_mAP = 0
    if args.resume:
        rank, mAP = reid_evaluator.evaluate(val_loader)
        best_rank1 = rank[0]
        best_mAP = mAP
        best_epoch = current_epoch + 1

    batch_time = AverageMeter()
    total_losses = AverageMeter()

    model_save_dir = os.path.join(args.save_dir, 'ckpts')
    os.makedirs(model_save_dir, exist_ok=True)

    summary_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tensorboard_log"),
                                   purge_step=global_step)

    def summary_loss(score, feat, labels, top_name='global'):
        loss = 0.0
        losses = loss_fn(score, feat, labels, mask=True if top_name == "local" else False)
        for loss_name, loss_val in losses.items():
            if loss_name.lower() == "accuracy":
                summary_writer.add_scalar(f"Score/{top_name}/triplet", loss_val, global_step)
                continue
            if "dist" in loss_name.lower():
                summary_writer.add_histogram(f"Distance/{loss_name}", loss_val, global_step)
                continue
            loss += loss_val
            summary_writer.add_scalar(f"losses/{top_name}/{loss_name}", loss_val, global_step)

        ohe_labels = torch.zeros_like(score)
        ohe_labels.scatter_(1, labels.unsqueeze(1), 1.0)

        cls_score = torch.softmax(score, dim=1)
        cls_score = torch.sum(cls_score * ohe_labels, dim=1).mean()
        summary_writer.add_scalar(f"Score/{top_name}/X-entropy", cls_score, global_step)

        return loss

    def save_weights(file_name, eph, steps):
        torch.save({"state_dict": model_state_dict,
                    "epoch": eph + 1,
                    "global_step": steps},
                   file_name)
        torch.save({"state_dict": optim_state_dict},
                   file_name.replace("model", "optimizer"))
        if args.center_loss:
            torch.save({"state_dict": center_state_dict},
                       file_name.replace("model", "optimizer_center"))
            torch.save({"state_dict": optim_state_dict},
                       file_name.replace("model", "center_param"))

    # training start
    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        t0 = time.time()
        for i, (inputs, labels, _, _) in enumerate(train_loader):
            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            cls_scores, features = model(inputs, labels)

            # losses
            total_loss = summary_loss(cls_scores[0], features[0], labels, 'global')
            if args.use_local_feat:
                total_loss += summary_loss(cls_scores[1], features[1], labels, 'local')

            optimizer.zero_grad()
            if args.center_loss:
                optimizer_center.zero_grad()

            # backward with global loss
            if args.amp:
                optimizers = [optimizer]
                if args.center_loss:
                    optimizers.append(optimizer_center)
                with amp.scale_loss(total_loss, optimizers) as scaled_loss:
                    scaled_loss.backward()
            else:
                with torch.autograd.detect_anomaly():
                    total_loss.backward()

            # optimization
            optimizer.step()
            if args.center_loss:
                for name, param in center_criterion.named_parameters():
                    try:
                        param.grad.data *= (1. / args.center_loss_weight)
                    except AttributeError:
                        continue
                optimizer_center.step()

            batch_time.update(time.time() - t0)
            total_losses.update(total_loss.item())

            # learning_rate
            current_lr = optimizer.param_groups[0]['lr']
            summary_writer.add_scalar("lr", current_lr, global_step)

            t0 = time.time()

            if (i + 1) % args.log_period == 0:
                print(f"Epoch: [{epoch}][{i+1}/{train_data_len}]  " +
                      f"Batch Time {batch_time.val:.3f} ({batch_time.mean:.3f})  " +
                      f"Total_loss {total_losses.val:.3f} ({total_losses.mean:.3f})")
            global_step += 1

        print(f"Epoch: [{epoch}]\tEpoch Time {batch_time.sum:.3f} s\tLoss {total_losses.mean:.3f}\tLr {current_lr:.2e}")
        # update learning rate
        scheduler.step()

        if args.eval_period > 0 and (epoch + 1) % args.eval_period == 0 or (epoch + 1) == args.max_epoch:
            rank, mAP = reid_evaluator.evaluate(val_loader)

            rank_string = ""
            for r in (1, 2, 4, 5, 8, 10, 16, 20):
                rank_string += f"Rank-{r:<3}: {rank[r-1]:.1%}"
                if r != 20:
                    rank_string += "    "
            summary_writer.add_text("Recall@K", rank_string, global_step)
            summary_writer.add_scalar("Rank-1", rank[0], (epoch+1))

            rank1 = rank[0]
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_mAP = mAP
                best_epoch = epoch + 1

            if (epoch + 1) % args.save_period == 0 or (epoch + 1) == args.max_epoch:
                pth_file_name = os.path.join(model_save_dir, f"{args.backbone}_model_{epoch + 1}.pth.tar")
                save_weights(pth_file_name, eph=epoch, steps=global_step)

            if is_best:
                pth_file_name = os.path.join(model_save_dir, f"{args.backbone}_model_best.pth.tar")
                save_weights(pth_file_name, eph=epoch, steps=global_step)

        # end epoch
        current_epoch += 1

        batch_time.reset()
        total_losses.reset()
        torch.cuda.empty_cache()

    print(f"Best rank-1 {best_rank1:.1%}, achived at epoch {best_epoch}")
    summary_writer.add_hparams({"dataset_name": args.dataset_name,
                                "triplet_dim": args.triplet_dim,
                                "margin": args.margin,

                                "base_lr": args.base_lr,

                                "use_attn": args.use_attn,
                                "use_mask": args.use_mask,
                                "use_local_feat": args.use_local_feat},

                               {"mAP": best_mAP,
                                "Rank1": best_rank1})


def main():
    args = argument_parsing()

    os.makedirs(args.save_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(args.save_dir, f"log_train({datetime.now()}).txt"))
    if args.cuda:
        if torch.cuda.is_available():
            cudnn.benchmark = True
        else:
            print("There is no available gpus!")
            args.cuda = False
    print("Running with {}s...".format("cpu" if args.cuda is False else "gpu"))

    print("user config".center(30, "="))
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
    print("end".center(30, "="))
    train(args)


if __name__ == "__main__":
    main()
