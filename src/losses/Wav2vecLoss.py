import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2vecLoss(nn.Module):
    def __init__(self, hf_path="facebook/wav2vec2-base-960h", **kwargs) -> None:
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(hf_path)
        self.model.eval()

        # 1️⃣  Freeze all model parameters (no grads w.r.t. them)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = None

    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        est: (B, C, T)
        gt: (B, C, T)
        """
        est = est.flatten(1,2)
        gt = gt.flatten(1,2)

        assert (torch.isnan(est).max() == 0), "Output tensor has nan!"
        assert (torch.isnan(gt).max() == 0), "GT tensor has nan!"

        if self.device != gt.device:
            # Move to CUDA
            self.model.to(gt.device)
            self.device = gt.device

        # 2️⃣  Embeddings of target: no gradient tracking needed
        with torch.no_grad():
            emb_Y = self.model(gt).last_hidden_state           # no graph, no memory

        # 3️⃣  Embeddings of prediction: keep graph so grads flow to Y_hat
        emb_Y_hat = self.model(est).last_hidden_state       # builds graph to Y_hat
        
        loss = torch.abs(emb_Y - emb_Y_hat).mean(dim=list(range(1, len(emb_Y.shape))))

        return loss

def test():
    x = torch.ones(2, 3, 10)
    y = x + torch.zeros(2, 3, 10)

    loss = Wav2vecLoss()
    print(loss(y, x))

if __name__ == "__main__":
    test()
