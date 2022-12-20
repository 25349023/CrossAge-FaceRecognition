from ModelFactory import model_insightface
import torch
from torchvision import transforms as trans


class insightFace():
    def __init__(self, mode):
        if mode == "mobilefacenet":
            self.model = model_insightface.MobileFaceNet(512).to('cpu')
            self.model.load_state_dict(torch.load('Model/model_mobilefacenet.pth', map_location=torch.device('cpu')))
        else:
            print("Wrong mode name for insighface")
            exit()
        self.model.eval()

    def extract_feat(self, face):
        transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        emb = self.model(transform(face).to('cpu').unsqueeze(0))
        return emb.detach().numpy()
