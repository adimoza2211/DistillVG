# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

import datasets.transforms as T
from .data_loader import SegVGDataset


def make_transforms(args, image_set, is_onestage=False):
    if is_onestage:
        normalize = Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return normalize

    imsize = args.imsize

    if image_set == 'train':
        scales = []
        if args.aug_scale:
            for i in range(7):
                scales.append(imsize - 32 * i)
        else:
            scales = [imsize]

        if args.aug_crop:
            crop_prob = 0.5
        else:
            crop_prob = 0.
    
        return T.Compose([
            T.RandomSelect(
                T.RandomResize(scales),
                T.Compose([
                    T.RandomResize([400, 500, 600], with_long_side=False),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales),
                ]),
                p=crop_prob
            ),
            T.ColorJitter(0.4, 0.4, 0.4),
            T.GaussianBlur(aug_blur=args.aug_blur),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize, aug_translate=args.aug_translate)
        ])


    if image_set in ['val', 'test', 'testA', 'testB']:
        return T.Compose([
            T.RandomResize([imsize]),
            T.ToTensor(),
            T.NormalizeAndPad(size=imsize),
        ])

    raise ValueError(f'unknown {image_set}')


def build_dataset(split, args):
    sgg_config = _build_sgg_config(args, split)
    return SegVGDataset(
        data_root=args.data_root,
        split_root=args.split_root,
        dataset=args.dataset,
        split=split,
        transform=make_transforms(args, split),
        max_query_len=args.max_query_len,
        sgg_config=sgg_config,
    )


def _build_sgg_config(args, split):
    enabled = getattr(args, 'sgg_enable', False)
    if not enabled:
        return None

    annotation_root = getattr(args, 'sgg_annotation_root', None)
    if annotation_root in (None, ''):
        annotation_root = ''

    config = {
        'enabled': True,
        'annotation_root': annotation_root,
        'num_edges': getattr(args, 'sgg_num_edges', 3),
        'relation_max': getattr(args, 'sgg_relation_max', 32),
        'aug_prob': getattr(args, 'sgg_aug_prob', 0.0),
        'eval_augmented_query': getattr(args, 'sgg_eval_augmented_query', False),
    }
    return config
