import datetime
import sys

import torch.optim
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms as T

from Evaluate import evaluate
from ModelFactory import FaceFeatureExtractor, model_insightface
from dataloader.dataset import CALFWDataset


if __name__ == '__main__':
    dataset = CALFWDataset('..\\data\\calfw', 'ForTraining\\CALFW_trainlist.csv',
                           transform=T.Compose([T.CenterCrop(112), T.ToTensor(),
                                                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    dataloader = DataLoader(dataset, 64, shuffle=True, num_workers=2, prefetch_factor=8)

    val_dataset = CALFWDataset(
        '..\\data\\calfw', 'ForTraining\\CALFW_validationlist.csv',
        transform=T.Compose([T.CenterCrop(112), T.ToTensor(),
                             T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    val_dataloader = DataLoader(val_dataset, 32, num_workers=2, prefetch_factor=8)

    model = FaceFeatureExtractor.insightFace("mobilefacenet", pretrained=False).model
    summary(model, [3, 112, 112])
    model.train().to('cuda')

    head = model_insightface.Arcface(embedding_size=512, classnum=dataset.num_class)
    head.train().to('cuda')

    cross_ent = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'weight_decay': 4e-5},
        {'params': head.kernel, 'weight_decay': 4e-4}
    ], lr=3e-3)

    for epoch in range(100):
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
        print(f'Epoch {epoch}: Loss = {loss:.6f}')

        model.eval()
        auc, r1_acc = evaluate(model, val_dataset, val_dataloader)
        print(f"\t| AUC: {auc:.3f}")
        print(f"\t| rank-1 ACC: {r1_acc:.3f}")

    torch.save(model.state_dict(), f'model-{datetime.datetime.now().strftime("%m%d-%H%M%S")}.pth')
