from torch.optim import SGD, Adam


def make_optimizer(args, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.base_lr
        weight_decay = args.weight_decay
        if "bias" in key:
            lr = args.base_lr * args.bias_lr_factor
            weight_decay = args.bias_weight_decay
        if ("feat_weight" in key) or ("gap_weight" in key) or ("gmp_weight" in key):
            lr = args.feat_weight_lr

        params += [{"params": [value], "lr": lr, "initial_lr": lr, "weight_decay": weight_decay}]
    if args.optimizer == 'sgd':
        optimizer = SGD(params, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = Adam(params, amsgrad=args.amsgrad)
    else:
        raise NotImplementedError()

    if center_criterion is not None:
        optimizer_center = SGD(center_criterion.parameters(), lr=args.center_loss_lr)
    else:
        optimizer_center = None

    return optimizer, optimizer_center
