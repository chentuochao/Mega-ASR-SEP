import torch
import torch.nn as nn

import src.utils as utils
from src.models.common.dsp import DualWindowTF
from src.models.common.dsp import DualWindowTF, MC_features_ONNX


# A TF-domain network guided by an embedding vector
class Net(nn.Module):
    def __init__(self,
                 number_of_microphones,
                 information_sharing,
                 current_model_name, current_model_params,
                 stft_chunk_size=64, stft_pad_size=32, stft_back_pad=32,
                 use_current_spfeats = False, use_masking=False, quantized = False):
        super(Net, self).__init__()
        self.use_masking = use_masking
        self.number_of_microphones = number_of_microphones
        self.information_sharing = information_sharing

        self.stft_module = DualWindowTF(stft_chunk_size=stft_chunk_size,
                                        stft_back_pad=stft_back_pad,
                                        stft_front_pad=stft_pad_size)
        # TF-Network
        if quantized:
            current_model_name = current_model_name.replace("src.models", "src.Qmodels")
            self.current_model_name = current_model_name
            
        self.current_tfnet = utils.import_attr(current_model_name)(n_fft=self.stft_module.nfft, **current_model_params)
        
        self.use_current_spfeats = use_current_spfeats
        
        self.nO = self.current_tfnet.n_srcs

    def init_buffers(self, batch_size, device):
        buffers = {}
        
        buffers['current_tfnet_bufs'] = self.current_tfnet.init_buffers(batch_size, device)
        buffers['istft_buf'] = self.stft_module.init_buffers(batch_size,  self.nO, device)

        return buffers

    def preprocess_audio(self, inputs, delay_chunks, input_state = None, pad=True):
        """
        mixture: (B, C, T)
        """
        xL = inputs['local_audio']
        xL = xL[:,:self.number_of_microphones,:]
        

        # Create empty state if it is not passed
        if input_state is None:
            input_state = self.init_buffers(xL.shape[0], xL.device)

        # STFT
        XL, _pad = self.stft_module.stft(xL, pad=pad) # [B, R, C, T, F]
        
        if self.information_sharing:
            xR = inputs['remote_audio']
            XR, _pad = self.stft_module.stft(xR, pad=pad) # [B, R, C, T, F]
            XL = torch.cat([XL, XR], dim=2)

        
        XL = XL.flatten(1,2) # [B, RC_local, T, F]

        return XL, _pad, input_state
    
    def postprocess_audio(self, XL, _pad, input_state):
        # Inverse STFT
        if 'istft_buf' in input_state:
            istft_state = input_state['istft_buf']
        x, input_state['istft_buf'] = self.stft_module.istft(XL, pad_amount=_pad, istft_buf=istft_state)

        return x, input_state


    def forward(self, inputs, delay_chunks, input_state = None, pad=True):
        """
        mixture: (B, C, T)
        """
        xL = inputs['local_audio']
        xL = xL[:,:self.number_of_microphones,:]
        

        # Create empty state if it is not passed
        if input_state is None:
            input_state = self.init_buffers(xL.shape[0], xL.device)

        # STFT
        XL, _pad = self.stft_module.stft(xL, pad=pad) # [B, R, C, T, F]
        
        if self.information_sharing:
            xR = inputs['remote_audio']
            XR, _pad = self.stft_module.stft(xR, pad=pad) # [B, R, C, T, F]
            XL = torch.cat([XL, XR], dim=2)

        XL = XL.flatten(1,2) # [B, RC_local, T, F]
        XL, input_state['current_tfnet_bufs'] = self.current_tfnet(XL, input_state = input_state['current_tfnet_bufs'])
        # Inverse STFT
        if 'istft_buf' in input_state:
            istft_state = input_state['istft_buf']
        x, input_state['istft_buf'] = self.stft_module.istft(XL, pad_amount=_pad, istft_buf=istft_state)

        return {'output': x, 'next_state': input_state}

    def compile(self):
        # Compile tf-gridnet only
        self.current_tfnet.forward = torch.compile(self.current_tfnet.forward)
        return self


if __name__ == "__main__":
    pass