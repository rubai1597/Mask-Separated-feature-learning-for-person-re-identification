import torch
from torch import nn

from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss


def make_loss(args, num_classes):
    feature_dim = args.triplet_dim

    if args.label_smooth:
        xent_criterion = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=args.cuda)
    else:
        xent_criterion = nn.CrossEntropyLoss()

    if args.triplet_loss:
        triplet_criterion = TripletLoss(margin=args.margin)

    if args.center_loss:
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feature_dim)
    else:
        center_criterion = None

    def loss_func(score, feat, target):
        losses = {}
        losses["X-entropy"] = xent_criterion(score, target)
        if feat is None:
            return losses

        if args.triplet_loss:
            tri_loss, dist_ap, dist_an = triplet_criterion(feat, target)
            losses["Triplet"] = args.triplet_loss_weight * tri_loss
            if args.margin == None:
                margin = 0.0
            else:
                margin = args.margin
            losses["Accuracy"] = (dist_an > dist_ap + margin).float().sum() / args.batch_size
            losses["dist_ap"] = dist_ap
            losses["dist_an"] = dist_an
        if args.center_loss:
            losses["Center"] = args.center_loss_weight * center_criterion(feat, target)
        return losses

    return loss_func, center_criterion
