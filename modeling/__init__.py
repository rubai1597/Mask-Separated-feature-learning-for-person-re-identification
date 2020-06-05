from .msreid import MSReID


def build_model(args, num_classes):
    if args.model_name == "msreid":
        model = MSReID(num_classes,
                         args.triplet_dim,
                         args.backbone_pretrain,
                         args.backbone,
                         args.use_attn,
                         args.use_mask,
                         args.use_local_feat)
    else:
        raise NotImplementedError()

    return model
