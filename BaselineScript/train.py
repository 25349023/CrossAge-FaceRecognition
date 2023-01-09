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

def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--schedule-lr', action='store_false', help='enable the lr scheduling')
    parser.add_argument('--schedule-epoch', type=int, default=20, help='decay the lr every N epochs')
    parser.add_argument('--schedule-step', type=float, default=0.3, help='decay the lr by the given factor',
                        metavar='DECAY_FACTOR')

    parser.add_argument('--crop', type=int, nargs=2, default=[231, 224],
                        help='two ints, the first is for CenterCrop, and the second is for Random Crop')
    parser.add_argument('--resize', type=int, default=112,
                        help='resize after cropping, 0 means no additional resize')
    parser.add_argument('--flip', type=float, default=0.5,
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
    # dataset = CALFWDataset(
    #     '../data/calfw', 'ForTraining/CALFW_trainlist_with_group.csv',
    #     transform=T.Compose(train_transform + [T.ToTensor(), normalization]), use_group=True)
    # dataloader = DataLoader(dataset, 64, shuffle=True, num_workers=2, prefetch_factor=8)

    dataset = CALFWDataset(
        '../data/calfw', 'ForTraining/aligned_aug_train.csv',
        transform=T.Compose(train_transform + [T.ToTensor(), normalization]), use_group=False, is_aug=True)
    dataloader = DataLoader(dataset, 64, shuffle=True, num_workers=2, prefetch_factor=8)

    val_dataset = CALFWDataset(
        '../data/calfw', 'ForTraining/CALFW_validationlist.csv',
        transform=T.Compose([T.CenterCrop(args.crop[0]), T.Resize(112), T.ToTensor(), normalization]))
    val_dataloader = DataLoader(val_dataset, 32, num_workers=2, prefetch_factor=8)

    model = model_insightface.GroupMobileFaceNet(512, True)
    model.load_state_dict(torch.load('./ckpt/model-best_gp_202.pth'))
    # model.load_gft_weight('./ckpt/model-best_nor_4.pth')
    # model = model_insightface.MobileFaceNet(512)
    # model.load_state_dict(torch.load('./ckpt/model-best_nor_512_101.pth'))
    model.train().to('cuda')
    # summary(model, [[3, 112, 112]])

    head = model_insightface.Arcface(embedding_size=512, classnum=dataset.num_class, m=.8, s=30)
    # head = model_insightface.Am_softmax(embedding_size=512, classnum=dataset.num_class, m=.35, s=30)
    head.train().to('cuda')

    cross_ent = nn.CrossEntropyLoss()
    Opt = getattr(torch.optim, args.optimizer)
    # optimizer = Opt([
    #     {'params': model.age_group.parameters()},
    #     {'params': model.adn.parameters()},
    #     {'params': model.id_fc.parameters()},
    # ], lr=args.lr)

    # optimizer1 = Opt([
    #     {'params': model.gfn.parameters(), 'lr':2e-6},
    #     {'params': head.kernel, 'weight_decay': 4e-4, 'lr':2e-4}
    # ], lr=args.lr)

    # bn_mod, nbn_mod = separate_bn_paras(model)

    # optimizer = Opt([
    #     {'params': nbn_mod[:-19], 'weight_decay': 4e-5, 'lr': 2e-6},
    #     {'params': nbn_mod[-19:] + [head.kernel], 'weight_decay': 4e-4},
    #     {'params': bn_mod},
    # ], lr=args.lr)

    optimizer = Opt([
        {'params': model.parameters()},
        {'params': head.kernel, 'weight_decay': 4e-4}
    ], lr=args.lr)

    # for param in model.gfn.parameters():
    #     param.requires_grad = False
    
    # for param in model.gfn.linear.parameters():
    #     param.requires_grad = True

    # for param in model.gfn.bn.parameters():
    #     param.requires_grad = True

    if args.schedule_lr:
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.schedule_epoch, args.schedule_step)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [i for i in range(8, 22, 4)], .5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, steps_per_epoch=len(dataloader)-1, epochs=args.epochs)

    best_R1 = None
    best_AUC = None
    best_loss = None
    best_epoch = None
    wait_time = 0
    for epoch in range(args.epochs):
        print()

        model.train()
        running_loss = 0
        for i, (x, y1) in enumerate(tqdm.tqdm(dataloader, file=sys.stdout, desc='Training: ', leave=False)):
            if x.shape[0] < 48:
                break
            x, y1 = x.to('cuda'), y1.to('cuda')
            # emb, age = model(x, True)
            emb = model(x)
            theta = head(emb, y1)
            loss1 = cross_ent(theta, y1)
            # loss2 = cross_ent(age, y2)
            loss = loss1 # + .1*loss2
            
            loss.backward()

            optimizer.step()
            # optimizer1.step()
            optimizer.zero_grad()
            # optimizer1.zero_grad()

            running_loss += loss.detach() * (x.shape[0] / len(dataset))

        print(f'Epoch {epoch}: Loss = {running_loss:.3f}')
        if args.schedule_lr:
            scheduler.step()

        
        if epoch % 1 == 0:
            model.eval()
            auc, r1_acc, far = evaluate(model, val_dataset, val_dataloader)
            print(f"\t| AUC: {auc*100:.2f}")
            print(f"\t| rank-1 ACC: {r1_acc*100:.2f}")
            far_string = str([f"{x*100:.2f}" for x in far]).replace('\'', '')
            print(f"\t| FAR {far_string}")

            if best_R1 is None or r1_acc > best_R1:
                wait_time = 0
                best_epoch = epoch
                best_R1 = r1_acc
                best_AUC = auc
                best_loss = running_loss
                pathlib.Path('ckpt').mkdir(exist_ok=True)
                ckpt_name = f'model-best.pth'
                torch.save(model.state_dict(), f'ckpt/{ckpt_name}')
            else:
                wait_time += 1

            if wait_time > 30:
                print("== early stop ==")
                break

    pathlib.Path('ckpt').mkdir(exist_ok=True)
    ckpt_name = f'model-{datetime.datetime.now().strftime("%m%d-%H%M%S")}.pth'
    torch.save(model.state_dict(), f'ckpt/{ckpt_name}')

    print('\n', f'Settings: {args}')
    print(f'Save checkpoint to ckpt/{ckpt_name}.')
    print(f'best epoch {best_epoch}, loss {best_loss:.3f}, best AUC {best_AUC*100:.2f}, best R1 {best_R1*100:.2f}')
