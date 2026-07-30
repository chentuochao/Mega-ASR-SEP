import torch
import torch.nn as nn
import torch.nn.functional as F

import src.utils as utils
from src.models.common.dsp import DualWindowTF, MC_features_ONNX

# A TF-domain network guided by an embedding vector
class Net(nn.Module):
    def __init__(self,
                 number_of_microphones,
                 current_model_name, current_model_params,
                 big_model_window_params,
                 delayed_model_name=None, delayed_model_params=None,
                 use_concat=False, use_masking=False, quantized=False,
                 use_delayed_spfeats = False, use_current_spfeats = False):
        super(Net, self).__init__()
        self.current_model_name = current_model_name
        self.delayed_model_name = delayed_model_name
        self.number_of_microphones = number_of_microphones
        self.use_delayed_spfeats = use_delayed_spfeats
        self.use_current_spfeats = use_current_spfeats
        
        self.use_masking = use_masking
        self.use_concat = use_concat
        self.big_frequency_layer_dimension = (big_model_window_params["stft_chunk_size"] + big_model_window_params["stft_back_pad"] + big_model_window_params["stft_pad_size"])//2 + 1 
        

        self.stft_module = DualWindowTF(stft_chunk_size=big_model_window_params["stft_chunk_size"],
                                            stft_back_pad=big_model_window_params["stft_back_pad"],
                                            stft_front_pad=big_model_window_params["stft_pad_size"])
        
        if quantized:
            current_model_name = current_model_name.replace("src.models", "src.Qmodels")
            self.current_model_name = current_model_name

        self.current_tfnet = utils.import_attr(current_model_name)(n_fft=self.stft_module.nfft, **current_model_params)
        self.nO = self.current_tfnet.n_srcs
        # TF-Network information-sharing
        if self.delayed_model_name:
            if quantized:
                delayed_model_name = delayed_model_name.replace("src.models", "src.Qmodels")
                self.delayed_model_name = delayed_model_name
            self.delayed_tfnet = utils.import_attr(delayed_model_name)(n_fft=self.stft_module.nfft, **delayed_model_params)
        
        
        print("#"*25)
        print("Single SE tfmlp ...")
        print(f"current model: {current_model_name}, delayed model: {delayed_model_name}")
        print(f"mics: {self.number_of_microphones}")
        print("Big_window_params:", big_model_window_params)
        print("Use masking:", use_masking)
        
        TotalParameters = 0
        TotalTrainabelParameters = 0
        total_params = sum(p.numel() for p in self.current_tfnet.parameters())
        trainable_params = sum(p.numel() for p in self.current_tfnet.parameters() if p.requires_grad)
        TotalParameters += total_params
        TotalTrainabelParameters += trainable_params
        print(f"Total Parameters: {total_params:,}, Trainable Parameters: {trainable_params:,}, (current_tfnet)")

        
        if self.delayed_model_name:
            total_params = sum(p.numel() for p in self.delayed_tfnet.parameters())
            trainable_params = sum(p.numel() for p in self.delayed_tfnet.parameters() if p.requires_grad)
            TotalParameters += total_params
            TotalTrainabelParameters += trainable_params
            print(f"Total Parameters: {total_params:,}, Trainable Parameters: {trainable_params:,}, (delayed_tfnet)")

        print(f"Total Parameters: {TotalParameters:,}, Trainable Parameters: {TotalTrainabelParameters:,}, (Total)")
        print("#"*25)


    def init_buffers(self, batch_size, device):
        buffers = {}
        
        buffers['current_tfnet_bufs'] = self.current_tfnet.init_buffers(batch_size, device)
        if self.delayed_model_name:
            buffers['delayed_tfnet_bufs'] = self.delayed_tfnet.init_buffers(batch_size, device)
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
        
        # if self.information_sharing:
        #     xR = inputs['remote_audio']
        #     XR, _pad = self.stft_module.stft(xR, pad=pad) # [B, R, C, T, F]
        #     XL = torch.cat([XL, XR], dim=2)

        
        XL = XL.flatten(1, 2) # [B, RC_local, T, F]

        return XL, _pad, input_state
    
    def postprocess_audio(self, XL, _pad, input_state):
        # Inverse STFT
        if 'istft_buf' in input_state:
            istft_state = input_state['istft_buf']
        x, input_state['istft_buf'] = self.stft_module.istft(XL, pad_amount=_pad, istft_buf=istft_state)

        return x, input_state


    def forward(self, inputs, delay_chunks, input_state = None, pad=True):
        """
        local_audio: (B, C, T)
        remote_audio: (B, C, T)
        """
        xL = inputs['local_audio']
        xL = xL[:,:self.number_of_microphones,:]
        
        if self.delayed_model_name:
            xR = inputs['remote_audio']
        
        
        if input_state is None:
            input_state = self.init_buffers(xL.shape[0], xL.device)        
        
        
        XL, _pad = self.stft_module.stft(xL, pad=True) # [B, R, C_L, T, F]
        if self.delayed_model_name:
            XR, _pad = self.stft_module.stft(xR, pad=True) # [B, R, C_R, T, F]
        if self.use_masking:
            big_reference_channel = XL[:,:,0,:,:]
        

        if delay_chunks > 0:
            padding = torch.zeros(XL.shape[0], XL.shape[1], XL.shape[2], delay_chunks, XL.shape[4], device=XL.device)
            if 'remote_context' in input_state:
                context_to_use = max(input_state['remote_context'].shape[3], delay_chunks)
                padding[..., -context_to_use:, :] = input_state['remote_context'][..., -context_to_use:, :]
            orig_steps = XL.shape[3]
            XL_delayed = torch.cat([padding, XL], dim=3)
            input_state['remote_context'] = XL_delayed[..., -delay_chunks:, :]
            
            XL_delayed = XL_delayed[..., :orig_steps, :]
        else:
            XL_delayed = XL

        if self.delayed_model_name:
            XD = torch.concatenate([XL_delayed, XR], dim=2) # [B, R, C_local + C_remote, T, F]
            ########################################################
            if self.use_delayed_spfeats:
                spfeats_delayed = MC_features_ONNX(XD[:, 0], XD[:, 1]) # [B, 3C - 3, T, F]
                #print("delayed: ", spfeats_delayed.shape)
                XD = XD.flatten(1,2) # [B, R(C_local + C_remote), T, F]
                XD = torch.cat([XD, spfeats_delayed], dim=1) # [B, R(C_local + C_remote) + spfeats, T, F]
            else:
                XD = XD.flatten(1,2) # [B, R(C_local + C_remote), T, F]
            
        if self.use_current_spfeats:
            
            spfeats_current = MC_features_ONNX(XL[:, 0], XL[:, 1]) # [B, 3C - 3, T, F]
            #print("current: ", spfeats_current.shape)
            XL = XL.flatten(1,2) # [B, RC_local, T, F]
            XL = torch.cat([XL, spfeats_current], dim=1) # [B, RC_local + spfeats, T, F]
        else:
            XL = XL.flatten(1,2) # [B, RC_local, T, F]
        ########################################################
        
        #print("Xl embedding shape:", XL_embedding.shape)
        
        if self.delayed_model_name: # information sharing            
            # Concatenate along channel dimension
            
            E, input_state['delayed_tfnet_bufs'] = self.delayed_tfnet(XD, input_state['delayed_tfnet_bufs'], conditioning_vector=None)
            #print("ash: ", E.shape, XL.shape, XD.shape)
            XL, input_state['current_tfnet_bufs'] = self.current_tfnet(XL, input_state['current_tfnet_bufs'], conditioning_vector=E)   
        
        else: # no information sharing 
            XL, input_state['current_tfnet_bufs'] = self.current_tfnet(XL, input_state['current_tfnet_bufs'], conditioning_vector=None)
            
            
        #print("Big model output: ", XL.shape)
        
        if self.use_masking:
            XL = big_reference_channel * XL
        
        if 'istft_buf' in input_state:
            big_istft_state = input_state['istft_buf']
        xL, input_state['istft_buf'] = self.stft_module.istft(XL, pad_amount=_pad, istft_buf=big_istft_state)

        return {'output': xL, 'next_state': input_state}

    def compile(self):
        # Compile tf-gridnet only
        self.current_tfnet.forward = torch.compile(self.current_tfnet.forward)
        if self.delayed_model_name:
            self.delayed_tfnet.forward = torch.compile(self.delayed_tfnet.forward)
        return self


if __name__ == "__main__":
    model_params= {
            "number_of_microphones":1,
            "use_masking": False,
            "big_model_window_params":{
                "stft_chunk_size": 128,
                "stft_pad_size": 64,
                "stft_back_pad": 64,
            },
            "delayed_model_name": "src.models.common.TFNet.tfnet.TFNet",
            "delayed_model_params": {
                "codec_channels": [
                    32
                ],
                "num_inputs": 4,
                "n_layers": 1,
                "return_decoded": False,
                "block_model_name": "src.models.blocks.mlpnet_block.MLPBlock",
                "block_model_params": {
                    "hidden_channels": 32,
                    "freq_compression": 1
                },
                # "conditioning_model_name": "src.models.common.film.FiLM",
                # "conditioning_model_params": {
                #     "embedding_channels": 0
                # },
		        "single_conditioning":True
            },
            "current_model_name": "src.models.common.TFNet.tfnet.TFNet",
            "current_model_params": {
                "codec_channels": [
                    32
                ],
                "num_inputs": 2,
                "n_srcs": 1,
                "n_layers": 6,
                "return_decoded": True,
                "block_model_name": "src.models.blocks.mlpnet_block.MLPBlock",
                "block_model_params": {
                    "hidden_channels": 32,
                    "freq_compression": 1
                },
                "conditioning_model_name": "src.models.common.film2.FiLM",
                "conditioning_model_params": {
                    "embedding_channels": 32
                },
		        "single_conditioning":True
            }
        }
    delay_chunks = 0
    Batch = 4
    Time = 16000
    Local_Mic = 3
    Remote_Mic = 1

    xL = torch.randn(Batch, Local_Mic, 5*Time)
    xR = torch.randn(Batch, Remote_Mic, 5*Time)
    
    inputs = {"local_audio": xL, 
                "remote_audio": xR
        }

    
    model = Net(**model_params)
    y = model(inputs, delay_chunks)["output"]
    print("x", xL.shape, " -> y:", y.shape)


