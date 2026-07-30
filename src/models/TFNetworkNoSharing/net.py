import torch
import torch.nn as nn

import src.utils as utils
from src.models.common.dsp import DualWindowTF
from src.models.common.dsp import DualWindowTF, MC_features_ONNX


# A TF-domain network guided by an embedding vector
class Net(nn.Module):
    def __init__(self,
                 current_model_name, current_model_params,
                 stft_chunk_size=64, stft_pad_size=32, stft_back_pad=32,
                 use_delayed_spfeats = False, use_current_spfeats = False,
                 use_masking=False, use_single_channel=False):
        super(Net, self).__init__()
        self.use_masking = use_masking
        self.use_single_channel = use_single_channel

        self.stft_module = DualWindowTF(stft_chunk_size=stft_chunk_size,
                                        stft_back_pad=stft_back_pad,
                                        stft_front_pad=stft_pad_size)
       
        # TF-Network
        self.current_tfnet = utils.import_attr(current_model_name)(n_fft=self.stft_module.nfft, **current_model_params)
        
        self.use_delayed_spfeats = use_delayed_spfeats
        self.use_current_spfeats = use_current_spfeats
        
        self.nO = self.current_tfnet.n_srcs

    def init_buffers(self, batch_size, device):
        buffers = {}
        
        buffers['current_tfnet_bufs'] = self.current_tfnet.init_buffers(batch_size, device)
        buffers['istft_buf'] = self.stft_module.init_buffers(batch_size,  self.nO, device)

        return buffers

    def forward(self, inputs, delay_chunks=0, input_state = None, pad=True):
        """
        mixture: (B, C, T)
        """
        xL = inputs['local_audio']

        # Create empty state if it is not passed
        if input_state is None:
            input_state = self.init_buffers(xL.shape[0], xL.device)

        # STFT
        XL, pad = self.stft_module.stft(xL, pad=pad) # [B, RC, T, F]

        if self.use_single_channel:
            XL = XL[:, :, :1]

        if self.use_current_spfeats:
            spfeats_current = MC_features_ONNX(XL[:, 0], XL[:, 1]) # [B, 3C - 3, T, F]
            XL = XL.flatten(1,2) # [B, RC_local, T, F]
            XL = torch.cat([XL, spfeats_current], dim=1) # [B, RC_local + spfeats, T, F]
        else:
            XL = XL.flatten(1,2) # [B, RC_local, T, F]

        XL, input_state['current_tfnet_bufs'] = self.current_tfnet(XL, input_state['current_tfnet_bufs'])

        # Inverse STFT
        if 'istft_buf' in input_state:
            istft_state = input_state['istft_buf']
        x, input_state['istft_buf'] = self.stft_module.istft(XL, pad_amount=pad, istft_buf=istft_state)

        return {'output': x, 'next_state': input_state}

    def compile(self):
        # Compile tf-gridnet only
        self.delayed_tfnet.forward = torch.compile(self.delayed_tfnet.forward)
        self.current_tfnet.forward = torch.compile(self.current_tfnet.forward)
        return self


if __name__ == "__main__":
    pass
