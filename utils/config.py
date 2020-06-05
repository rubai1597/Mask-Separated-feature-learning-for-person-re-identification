from argparse import ArgumentParser

from utils.common import readable_directory, positive_int, nonnegative_int, positive_float, probability


def argument_parsing():
    parser = ArgumentParser(description="ReID Model Training/Testing")

    # basic setting
    parser.add_argument("--cuda", default=False, action="store_true")
    parser.add_argument("--amp", default=False, action="store_true")

    # data transformation/augmentation
    # resize -> horizontal flip -> random crop -> normalize -> random erasing
    parser.add_argument("--input_size", default=[256, 128], type=positive_int, nargs=2,
                        help="The size of a network.")
    parser.add_argument("--h_flip", default=0.5, type=probability,
                        help="The probability of horizontal flipping.")
    parser.add_argument("--input_pad", default=10, type=nonnegative_int,
                        help="The padding value which is used in random cropping.")
    parser.add_argument("--norm_mean", default=[0.485, 0.456, 0.406], type=positive_float, nargs=3,
                        help="The mean values to normalize the input image. It is commonly set to be ImageNet constant.")
    parser.add_argument("--norm_std", default=[0.229, 0.224, 0.225], type=positive_float, nargs=3,
                        help="The std values to normalize the input image. It is commonly set to be ImageNet constant.")
    parser.add_argument("--re_prob", default=0.5, type=probability,
                        help="The probability of random erasing.")

    # data loader
    parser.add_argument("--dataset_name", default='market1501', type=str, choices=['market1501', 'dukemtmc',
                                                                                   'cuhk03-D', 'cuhk03-L'])
    parser.add_argument("--dataset_root", default='/workdir/jinbeom/dataset/reid', type=readable_directory,
                        help="The root directory of datasets.\n"
                             "Data would be loaded with os.path.join(args.dataset_root, args.dataset_name)")
    parser.add_argument("--num_workers", default=8, type=positive_int)
    parser.add_argument("--batch_size", default=96, type=positive_int)
    parser.add_argument("--num_instance", default=3, type=positive_int)

    # model settings
    parser.add_argument("--model_name", default='msreid', type=str, choices=['msreid'])
    parser.add_argument("--backbone", default='resnet50', type=str, choices=['resnet50'])
    parser.add_argument("--backbone_pretrain", default='./resnet50-19c8e357.pth')
    parser.add_argument("--triplet_dim", default=1536, type=positive_int)

    parser.add_argument("--use_attn", default=False, action="store_true")
    parser.add_argument("--use_mask", default=False, action="store_true")
    parser.add_argument("--use_local_feat", default=False, action="store_true")

    # training settings
    parser.add_argument("--save_dir", default='outputs', type=str)
    parser.add_argument("--pretrain", default='', type=str)
    parser.add_argument("--resume", default=False, action="store_true")

    parser.add_argument("--optimizer", default='adam', type=str, choices=['sgd', 'adam'])
    parser.add_argument("--amsgrad", default=False, action="store_true")
    parser.add_argument("--momentum", default=0.9, type=positive_float)

    parser.add_argument("--base_lr", default=1.5e-3, type=positive_float)
    parser.add_argument("--bias_lr_factor", default=1.0, type=positive_float)
    parser.add_argument("--weight_decay", default=5e-4, type=positive_float)
    parser.add_argument("--bias_weight_decay", default=5e-4, type=positive_float)

    parser.add_argument("--triplet_loss", default=False, action="store_true")
    parser.add_argument("--triplet_loss_weight", default=1.0, type=positive_float)
    parser.add_argument("--margin", default=None, type=positive_float)
    parser.add_argument("--center_loss", default=False, action="store_true")
    parser.add_argument("--center_loss_weight", default=5e-4, type=positive_float)
    parser.add_argument("--center_loss_lr", default=0.5, type=positive_float)
    parser.add_argument("--label_smooth", default=False, action="store_true")

    parser.add_argument("--max_epoch", default=180, type=positive_int)
    parser.add_argument("--steps", default=[60, 120], type=positive_int, nargs="+")
    parser.add_argument("--warmup_step", default=10, type=nonnegative_int)
    parser.add_argument("--warmup_factor", default=0.1, type=positive_float)
    parser.add_argument("--gamma", default=0.1, type=float)

    parser.add_argument("--log_period", default=40, type=positive_int)
    parser.add_argument("--eval_period", default=2, type=positive_int)
    parser.add_argument("--save_period", default=40, type=positive_int)
    parser.add_argument("--test_norm", default=False, action="store_true")

    return parser.parse_args()


args = argument_parsing()
