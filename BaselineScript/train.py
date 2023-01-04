import argparse
import datetime
import os
import pathlib
import sys
from typing import Any, List

import torch.optim
import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms as T

from Evaluate import evaluate
from ModelFactory import FaceFeatureExtractor, model_insightface
from dataloader.dataset import CALFWDataset, PairCALFWDataset


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--schedule-lr', action='store_true', help='enable the lr scheduling')
    parser.add_argument('--schedule-epoch', type=int, default=6, help='decay the lr every N epochs')
    parser.add_argument('--schedule-step', type=float, default=0.3, help='decay the lr by the given factor',
                        metavar='DECAY_FACTOR')

    parser.add_argument('--crop', type=int, nargs=2, default=[112, 112],
                        help='two ints, the first is for CenterCrop, and the second is for Random Crop')
    parser.add_argument('--resize', type=int, default=0,
                        help='resize after cropping, 0 means no additional resize')
    parser.add_argument('--flip', type=float, default=0,
                        help='if greater than 0, enable the random horizontal flipping')
    parser.add_argument('--rotate', type=float, default=0,
                        help='if greater than 0, enable the random rotation', metavar='DEG')

    parser.add_argument('--jitter', action='store_true', help='enable the color jittering')
    parser.add_argument('--jitter-param', type=float, default=[0.125, 0.125, 0.125, 0], nargs=4,
                        help='settings for Color Jittering',
                        metavar=('BRIGHTNESS', 'CONTRAST', 'SATURATION', 'HUE'))

    parser.add_argument('--head', default='Arcface', help='set the head module')

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    print(args, '\n')

    train_transform: List[Any] = [T.CenterCrop(args.crop[0]), T.RandomCrop(args.crop[1])]
    if args.resize > 0:
        train_transform.append(T.Resize(args.resize))

    if args.flip > 0:
        train_transform.append(T.RandomHorizontalFlip(args.flip))

    if args.rotate > 0:
        train_transform.append(T.RandomRotation(args.rotate))

    if args.jitter:
        train_transform.append(T.ColorJitter(*args.jitter_param))

    normalization = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    dataset = CALFWDataset(
        '../data/calfw', 'ForTraining/CALFW_trainlist.csv',
        transform=T.Compose(train_transform + [T.ToTensor(), normalization]))
    dataloader = DataLoader(dataset, 64, shuffle=True, num_workers=2, prefetch_factor=8)

    pair_dataset = PairCALFWDataset(
        '../data/calfw', 'ForTraining/CALFW_trainlist.csv',
        transform=T.Compose(train_transform + [T.ToTensor(), normalization]))
    pair_dataloader = DataLoader(pair_dataset, 64, shuffle=True, num_workers=2, prefetch_factor=8)

    val_dataset = CALFWDataset(
        '../data/calfw', 'ForTraining/CALFW_validationlist.csv',
        transform=T.Compose([T.CenterCrop(112), T.ToTensor(), normalization]))
    val_dataloader = DataLoader(val_dataset, 32, num_workers=2, prefetch_factor=8)

    model = FaceFeatureExtractor.insightFace("mobilefacenet", ckpt_path=False).model
    model.train().to('cuda')
    summary(model, [[3, 112, 112]])

    Head = getattr(model_insightface, args.head)
    head = Head(embedding_size=512, classnum=dataset.num_class)
    head.train().to('cuda')

    similarity = model_insightface.SinCosCrossSim(embedding_size=512, classnum=dataset.num_class)
    similarity.train().to('cuda')

    cross_ent = nn.CrossEntropyLoss()
    Opt = getattr(torch.optim, args.optimizer)
    optimizer = Opt([
        {'params': model.parameters()},
        {'params': head.kernel, 'weight_decay': 4e-4}
    ], lr=args.lr)

    if args.schedule_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.schedule_epoch, args.schedule_step)

    for epoch in range(args.epochs):
        print()

        model.train()
        for x, y in tqdm.tqdm(dataloader, file=sys.stdout, desc='Training: ', leave=False):
            x, y = x.to('cuda'), y.to('cuda')
            emb = model(x)
            theta = head(emb, y)
            loss = cross_ent(theta, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for x1, x2 in tqdm.tqdm(pair_dataloader, file=sys.stdout, desc='Training: ', leave=False):
            x1, x2 = x1.to('cuda'), x2.to('cuda')
            emb_pos = (model(x1), model(x2))
            emb_neg = (emb_pos[0], emb_pos[1].roll(1, dims=0))
            # emb2 = [em.roll(1, dims=0) for em in emb1]
            # cos_loss = similarity(emb1, emb2)
            pos_cos_loss = -F.cosine_similarity(*emb_pos).mean()
            neg_cos_loss = F.cosine_similarity(*emb_neg).mean()
            cos_loss = 0.5 * pos_cos_loss + 0.5 * neg_cos_loss

            optimizer.zero_grad()
            cos_loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}: Loss = {loss:.6f}, '
              f'Cos similarity (P) = {-pos_cos_loss:.2f}, (N) = {neg_cos_loss:.2f}')

        if args.schedule_lr:
            scheduler.step()

        model.eval()
        auc, r1_acc = evaluate(model, val_dataset, val_dataloader)
        print(f"\t| AUC: {auc:.3f}")
        print(f"\t| rank-1 ACC: {r1_acc:.3f}")

    pathlib.Path('ckpt').mkdir(exist_ok=True)
    ckpt_name = f'model-{datetime.datetime.now().strftime("%m%d-%H%M%S")}.pth'
    torch.save(model.state_dict(), f'ckpt/{ckpt_name}')

    print('\n', f'Settings: {args}')
    print(f'Save checkpoint to ckpt/{ckpt_name}.')
