import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)

class BiologicallyPlausibleNerfModel(nn.Module):
    def __init__(self, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):   
        super(BiologicallyPlausibleNerfModel, self).__init__()
        
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        # density estimation
        self.block1_origin = nn.Sequential(nn.Linear(input_ch, W), nn.ReLU(),
                                    nn.Linear(W, W), nn.ReLU(),
                                    nn.Linear(W, W), nn.ReLU(), )
        
        self.block2_origin = nn.Sequential(nn.Linear(input_ch + W, W), nn.ReLU(),
                                    nn.Linear(W, W), nn.ReLU(), 
                                    nn.Linear(W, 1), )
        
        
        self.pre_mod = nn.Linear(W*2, W)
        self.f_mem_o = nn.Linear(W*2, W)
        self.f_mem_d = nn.Linear(W*2, W)
        self.memory = None
        
        # color estimation
        self.block1_direction = nn.Sequential(nn.Linear(input_ch, W), nn.ReLU(),
                                    nn.Linear(W, W), nn.ReLU(), 
                                    nn.Linear(W, W), nn.ReLU(), )
        
        self.block2_direction = nn.Sequential(nn.Linear(input_ch_views+ W, W//2), nn.ReLU(),
                                    nn.Linear(W//2, 3),  )

        self.relu = nn.ReLU()


    def forward(self, x):
        # o: origins
        # d: direction

        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts

        h_o = self.block1_origin(input_pts) # h: [batch_size, hidden_dim]
        h_d = self.block1_direction(input_pts) # h: [batch_size, hidden_dim]


        # memory
        f_o = torch.sigmoid(h_o)
        f_d = torch.sigmoid(h_d)
        h_o_h_d = torch.cat((h_o, h_d), dim=1)
        pre_mod = torch.tanh(self.pre_mod(h_o_h_d))
        
        f_mod = torch.sigmoid(self.f_mem_d(h_o_h_d))
        mod = torch.mul(pre_mod, f_mod)

        if self.memory is None:
            memory = torch.tanh(mod)
            self.memory = memory.detach()
        else:
            if self.memory.shape[0]!=input_pts.shape[0]:
                f_mem = torch.sigmoid(self.f_mem_o(h_o_h_d))

                memory = torch.tanh(mod + torch.mul(f_mem, self.memory[:input_pts.shape[0],:]))
                new_memory = self.memory.clone()
                new_memory[:memory.shape[0],:] = memory
                self.memory = new_memory.detach()

                del new_memory      
                torch.cuda.empty_cache()    


            else:
                f_mem = torch.sigmoid(self.f_mem_o(h_o_h_d))
                memory = torch.tanh(mod + torch.mul(f_mem, self.memory))
                self.memory = memory.detach()    

        # block 2
        sigma = self.block2_origin(torch.cat((torch.mul(f_o, memory), input_pts), dim=1)) # h: [batch_size, hidden_dim // 2]
        c =  self.block2_direction(torch.cat((torch.mul(f_d, memory), input_views), dim=1))  # h: [batch_size, hidden_dim // 2]


        del memory
        torch.cuda.empty_cache()
        outputs = torch.cat([c, sigma], -1)
        return outputs