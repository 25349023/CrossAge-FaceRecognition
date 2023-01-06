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
    parser.add_argument('--epochs', type=int, default=40, help=' ')
    parser.add_argument('--lr', type=float, default=3e-3, help=' ')
    parser.add_argument('--optimizer', default='Adam', help=' ')
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

    parser.add_argument('--contrastive', action='store_true',
                        help='enable supervised contrastive loss')
    parser.add_argument('--cont-factor', type=float, default=1.0,
                        help='factor of contrastive loss')
    parser.add_argument('--cont-intra', action='store_true', help='enable intra contrastive loss')

    parser.add_argument('--freeze-head', type=int, default=0, help='freeze the head after N epochs')
    parser.add_argument('--refresh-head', type=int, default=0, help='re-initialize the head after N epochs')

    parser.add_argument('--ckpt', default='', help='continue training')

    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    if all([args.freeze_head, args.refresh_head]):
        raise ValueError('cannot specify freeze-head and refresh-head at the same time')

    print(args, '\n')

    train_transform: List[Any] = [T.Resize(args.crop[0]), T.RandomCrop(args.crop[1])]
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
        transform=T.Compose([T.Resize(112), T.ToTensor(), normalization]))
    val_dataloader = DataLoader(val_dataset, 32, num_workers=2, prefetch_factor=8)

    model = FaceFeatureExtractor.insightFace("mobilefacenet", ckpt_path=args.ckpt).model
    model.train().to('cuda')
    summary(model, [[3, 112, 112]])

    if args.ckpt:
        print(f' Continue training from {args.ckpt} '.center(80, '#'))

    Head = getattr(model_insightface, args.head)
    head = Head(embedding_size=512, classnum=dataset.num_class)
    head.train().to('cuda')

    cross_ent = nn.CrossEntropyLoss()
    contrastive_loss = model_insightface.ContrastiveLoss()
    Opt = getattr(torch.optim, args.optimizer)
    optimizer = Opt([
        {'params': model.parameters(), 'weight_decay': 1e-4},
        {'params': head.kernel, 'weight_decay': 1e-3, 'lr': args.lr * 0.5}
    ], lr=args.lr)

    if args.schedule_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.schedule_epoch, args.schedule_step)

    pathlib.Path('ckpt').mkdir(exist_ok=True)
    start_time = datetime.datetime.now().strftime("%m%d-%H%M%S")
    best_acc, best_ckpt_name = 0, f'model-best-{start_time}.pth'

    for epoch in range(args.epochs):
        print()

        if 0 < args.freeze_head == epoch:
            print(' freeze the head '.center(30, '='))
            head.kernel.requires_grad_(False)

        if 0 < args.refresh_head == epoch:
            print(' refresh the head '.center(30, '='))
            head.init_kernel()

        model.train()
        for x, y in tqdm.tqdm(dataloader, file=sys.stdout, desc='Training: ', leave=False):
            x, y = x.to('cuda'), y.to('cuda')
            emb = model(x)
            theta = head(emb, y)
            loss = cross_ent(theta, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if args.contrastive:
            for x1, x2 in tqdm.tqdm(pair_dataloader, file=sys.stdout, desc='Training (Cont): ', leave=False):
                x1, x2 = x1.to('cuda'), x2.to('cuda')
                emb = (model(x1), model(x2))
                inter_loss, intra_loss = contrastive_loss(*emb)
                ct_loss = inter_loss
                if args.cont_intra:
                    ct_loss += intra_loss
                ct_loss *= args.cont_factor

                optimizer.zero_grad()
                ct_loss.backward()
                optimizer.step()

        print(f'Epoch {epoch}: Loss = {loss:.6f}')
        if args.contrastive:
            print(f'\t| Contrastive loss = {ct_loss:.2f}')

        if args.schedule_lr:
            scheduler.step()

        model.eval()
        auc, r1_acc = evaluate(model, val_dataset, val_dataloader)
        if r1_acc > best_acc:
            best_acc = r1_acc
            torch.save(model.state_dict(), f'ckpt/{best_ckpt_name}')

        print(f"\t| AUC: {auc:.3f}")
        print(f"\t| rank-1 ACC: {r1_acc:.3f}")

    ckpt_name = f'model-{start_time}.pth'
    torch.save(model.state_dict(), f'ckpt/{ckpt_name}')

    print('\n', f'Settings: {args}')
    print(f'Save checkpoint to ckpt/{ckpt_name}.')
    print(f'Save best checkpoint to ckpt/{best_ckpt_name}, best acc = {best_acc}.')
