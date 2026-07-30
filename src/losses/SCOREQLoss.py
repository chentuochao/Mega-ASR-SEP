import torch
import torch.nn as nn

import os
from urllib.request import urlretrieve
from tqdm import tqdm


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(n - self.n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

# PyTorch classes needed for the use_onnx=False fallback
class TripletModel(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim, emb_dim=256):
        super(TripletModel, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.embedding_layer = nn.Sequential(nn.ReLU(), nn.Linear(self.ssl_features, emb_dim))
    
    def forward(self, wav, phead=False):
        wav = wav.squeeze(1)
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x = torch.mean(x, 1)
        if phead:
            x = self.embedding_layer(x)
        x = torch.nn.functional.normalize(x, dim=1)
        return x

class MosPredictor(nn.Module):
    def __init__(self, pt_model, emb_dim=768):
        super(MosPredictor, self).__init__()
        self.pt_model = pt_model
        self.mos_layer = nn.Linear(emb_dim, 1)
        
    def forward(self, wav):
        x = self.pt_model(wav, phead=False)
        if len(x.shape) == 3: x.squeeze_(2)
        out = self.mos_layer(x)
        return out

class SCOREQLoss(nn.Module):
    def __init__(self, data_domain='natural', sr=16000) -> None:
        super().__init__()
        self.data_domain = data_domain
        self.device = None

        self._init_pytorch()

    def _init_pytorch(self):
        """Initializes the original PyTorch/fairseq model."""
        try:
            import fairseq
        except ImportError:
            raise ImportError(
                "PyTorch/fairseq mode requires 'fairseq' and 'torch'. "
                "Please install them with: pip install scoreq[pytorch]"
            )
        
        print("Initializing in PyTorch mode. `fairseq` and `torch` are required.")
        if torch.cuda.is_available(): self.device = 'cuda'
        else: self.device = 'cpu'

        url_w2v = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt"
        CHECKPOINT_PATH = self._download_model("wav2vec_small.pt", url_w2v, "pt-models")
        
        # Temporarily monkey-patch torch.load to default to weights_only=False.
        # This is necessary because fairseq's internal loading function does not
        # expose this argument, and it's required for newer PyTorch versions to
        # load old checkpoints containing non-tensor data.
        original_torch_load = torch.load
        try:
            def new_torch_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            
            torch.load = new_torch_load
            
            w2v_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([CHECKPOINT_PATH])
        finally:
            torch.load = original_torch_load
        
        ssl_model = w2v_model[0]
        ssl_model.remove_pretraining_modules()

        model = TripletModel(ssl_model, ssl_out_dim=768, emb_dim=256)
            
        PT_URLS = {
            ('natural', 'nr'): 'https://zenodo.org/records/13860326/files/adapt_nr_telephone.pt',
            ('natural', 'ref'): 'https://zenodo.org/records/13860326/files/fixed_nmr_telephone.pt',
            ('synthetic', 'nr'): 'https://zenodo.org/records/13860326/files/adapt_nr_synthetic.pt',
            ('synthetic', 'ref'): 'https://zenodo.org/records/13860326/files/fixed_nmr_synthetic.pt',
        }
        model_key = (self.data_domain, 'ref')#(self.data_domain, self.mode)
        model_url = PT_URLS.get(model_key)
        if not model_url:
            raise ValueError(f"Invalid model combination: domain='{self.data_domain}', mode='{self.mode}'")
        
        MODEL_PATH = self._download_model(os.path.basename(model_url), model_url, "pt-models")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=False))
        
        self.model = model
        self.model.eval()

    def _download_model(self, filename, url, cache_dir_name):
        """Helper to download a model from a URL with a progress bar."""
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "scoreq", cache_dir_name)
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, filename)

        if not os.path.exists(model_path):
            print(f"Downloading {filename}...")
            try:
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                    urlretrieve(url, model_path, reporthook=t.update_to)
                print("Download complete.")
            except Exception as e:
                print(f"Error downloading model: {e}")
                if os.path.exists(model_path): os.remove(model_path)
                raise e

        return model_path

    def forward(self, est: torch.Tensor, gt: torch.Tensor, *args, **kwargs):
        """
        est, gt: [B, C, t]
        """
        B, C, t = gt.shape

        if self.device != gt.device:
            # Move to CUDA
            self.model.to(gt.device)
            self.device = gt.device
        
        # L2 Norm
        scoreq_loss = torch.linalg.norm(self.model(est.flatten(0,1)) - self.model(gt.flatten(0,1)), dim=-1)
        
        return scoreq_loss.reshape(B, C)
    
if __name__ == "__main__":
    loss = SCOREQLoss()

    x = torch.randn(2, 3, 10000)
    y = torch.randn(2, 3, 10000)

    z = loss(x, y)

    print(z)