from ModelFactory import model_insightface
import torch
from torchvision import transforms as trans


class insightFace:
    def __init__(self, mode, ckpt_path=None, sigma=1):
        if mode == "mobilefacenet":
            self.model = model_insightface.MobileFaceNet(512, sigma).to('cpu')
            if ckpt_path:
                self.model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
        else:
            raise ValueError("Wrong mode name for insighface")
        self.model.eval()

    def extract_feat(self, face):
        transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        emb = self.model(transform(face).to('cpu').unsqueeze(0))
        return emb.detach().numpy()
