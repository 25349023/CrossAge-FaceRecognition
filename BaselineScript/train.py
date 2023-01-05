import argparse
import datetime
import os
import pathlib
import sys
from typing import Any, List

import torch.optim
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms as T

from Evaluate import evaluate
from ModelFactory import FaceFeatureExtractor, model_insightface
from dataloader.dataset import CALFWDataset


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--schedule-lr', action='store_false', help='enable the lr scheduling')
    parser.add_argument('--schedule-epoch', type=int, default=30, help='decay the lr every N epochs')
    parser.add_argument('--schedule-step', type=float, default=0.3, help='decay the lr by the given factor',
                        metavar='DECAY_FACTOR')

    parser.add_argument('--crop', type=int, nargs=2, default=[231, 231],
                        help='two ints, the first is for CenterCrop, and the second is for Random Crop')
    parser.add_argument('--resize', type=int, default=112,
                        help='resize after cropping, 0 means no additional resize')
    parser.add_argument('--flip', type=float, default=.5,
                        help='if greater than 0, enable the random horizontal flipping')
    parser.add_argument('--rotate', type=float, default=0,
                        help='if greater than 0, enable the random rotation', metavar='DEG')

    parser.add_argument('--jitter', action='store_false', help='enable the color jittering')
    parser.add_argument('--jitter-param', type=float, default=[0.125, 0.125, 0.4, 0], nargs=4,
                        help='settings for Color Jittering',
                        metavar=('BRIGHTNESS', 'CONTRAST', 'SATURATION', 'HUE'))

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
        transform=T.Compose(train_transform + [T.ToTensor(), normalization]), use_group=False)
    dataloader = DataLoader(dataset, 64, shuffle=True, num_workers=2, prefetch_factor=8)

    val_dataset = CALFWDataset(
        '../data/calfw', 'ForTraining/CALFW_validationlist.csv',
        transform=T.Compose([T.Resize(112), T.ToTensor(), normalization]))
    val_dataloader = DataLoader(val_dataset, 32, num_workers=2, prefetch_factor=8)

    # model = model_insightface.GroupMobileFaceNet(512)
    model = model_insightface.MobileFaceNet(512)
    model.train().to('cuda')
    summary(model, [[3, 112, 112]])

    # head = model_insightface.Arcface(embedding_size=512, classnum=dataset.num_class)
    head = model_insightface.Am_softmax(embedding_size=512, classnum=dataset.num_class)
    head.train().to('cuda')

    cross_ent = nn.CrossEntropyLoss()
    Opt = getattr(torch.optim, args.optimizer)
    optimizer = Opt([
        {'params': model.parameters()},
        {'params': head.kernel, 'weight_decay': 4e-4}
    ], lr=args.lr)

    if args.schedule_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.schedule_epoch, args.schedule_step)

    best_R1 = None
    best_AUC = None
    best_loss = None
    wait_time = 0
    for epoch in range(args.epochs):
        print()

        model.train()
        running_loss = 0
        for x, y1 in tqdm.tqdm(dataloader, file=sys.stdout, desc='Training: ', leave=False):
            x, y1 = x.to('cuda'), y1.to('cuda')
            # emb, age = model(x)
            emb = model(x)
            theta = head(emb, y1)
            loss1 = cross_ent(theta, y1)
            # loss2 = cross_ent(age, y2)
            loss = loss1 # + .1*loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach() * (x.shape[0] / len(dataset))

        print(f'Epoch {epoch}: Loss = {running_loss:.6f}')
        if args.schedule_lr:
            scheduler.step()

        model.eval()
        auc, r1_acc = evaluate(model, val_dataset, val_dataloader)
        print(f"\t| AUC: {auc:.3f}")
        print(f"\t| rank-1 ACC: {r1_acc:.3f}")

        if best_R1 is None or r1_acc > best_R1:
            wait_time = 0
            best_R1 = r1_acc
            best_AUC = auc
            best_loss = loss
            pathlib.Path('ckpt').mkdir(exist_ok=True)
            ckpt_name = f'model-best.pth'
            torch.save(model.state_dict(), f'ckpt/{ckpt_name}')
        else:
            wait_time += 1

        if wait_time > 20:
            print("== early stop ==")
            break

    pathlib.Path('ckpt').mkdir(exist_ok=True)
    ckpt_name = f'model-{datetime.datetime.now().strftime("%m%d-%H%M%S")}.pth'
    torch.save(model.state_dict(), f'ckpt/{ckpt_name}')

    print('\n', f'Settings: {args}')
    print(f'Save checkpoint to ckpt/{ckpt_name}.')
    print(f'best AUC {best_AUC:.3f}, best R1 {best_R1:.3f}, best loss {best_loss:.6f}')
