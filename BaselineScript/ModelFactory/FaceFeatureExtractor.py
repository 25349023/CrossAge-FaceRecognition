from ModelFactory import model_insightface
import torch
from torchvision import transforms as trans


class insightFace():
    def __init__(self, mode, ckpt_path=None):
        if mode == "mobilefacenet":
            self.model = model_insightface.MobileFaceNet(512).to('cpu')
            if ckpt_path:
                try:
                    self.model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
                except:
                    print(f"error to read model file: {ckpt_path}")
                    exit()
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
