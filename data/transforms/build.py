from torchvision import transforms as T


def build_transforms(args, is_train=True):
    if is_train:
        return T.Compose([
            T.Resize(args.input_size),
            T.RandomHorizontalFlip(args.h_flip),
            T.Pad(args.input_pad),
            T.RandomCrop(args.input_size),
            T.ToTensor(),
            T.Normalize(mean=args.norm_mean, std=args.norm_std),
            T.RandomErasing(p=args.re_prob, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=(0.485, 0.456, 0.406))
        ])
    else:
        return T.Compose([
            T.Resize(args.input_size),
            T.ToTensor(),
            T.Normalize(mean=args.norm_mean, std=args.norm_std)
        ])
