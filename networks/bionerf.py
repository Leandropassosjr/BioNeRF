import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)

class BiologicallyPlausibleNerfModel(nn.Module):
    def __init__(self, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):   
        super(BiologicallyPlausibleNerfModel, self).__init__()
        
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        # density estimation
        self.M_delta = nn.Sequential(nn.Linear(input_ch, W), nn.ReLU(),
                                    nn.Linear(W, W), nn.ReLU(),
                                    nn.Linear(W, W), nn.ReLU(), )
        
        self.M_line_delta = nn.Sequential(nn.Linear(input_ch + W, W), nn.ReLU(),
                                    nn.Linear(W, W), nn.ReLU(), 
                                    nn.Linear(W, 1), )
        
        
        self.W_gamma = nn.Linear(W*2, W)
        self.W_psi = nn.Linear(W*2, W)
        self.W_mu = nn.Linear(W*2, W)
        self.memory = None
        
        # color estimation
        self.M_c = nn.Sequential(nn.Linear(input_ch, W), nn.ReLU(),
                                    nn.Linear(W, W), nn.ReLU(), 
                                    nn.Linear(W, W), nn.ReLU(), )
        
        self.M_line_c = nn.Sequential(nn.Linear(input_ch_views+ W, W//2), nn.ReLU(),
                                    nn.Linear(W//2, 3),  )


    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts

        h_delta = self.M_delta(input_pts) # h: [batch_size, hidden_dim]
        h_c = self.M_c(input_pts) # h: [batch_size, hidden_dim]


        # memory
        f_delta = torch.sigmoid(h_delta)
        f_c = torch.sigmoid(h_c)
        h_delta_h_c = torch.cat((h_delta, h_c), dim=1)
        gamma = torch.tanh(self.W_gamma(h_delta_h_c))
        
        f_mu = torch.sigmoid(self.W_mu(h_delta_h_c))
        mu = torch.mul(gamma, f_mu)

        if self.memory is None:
            memory = torch.tanh(mu)
            self.memory = memory.detach()
        else:
            if self.memory.shape[0]!=input_pts.shape[0]:
                f_psi = torch.sigmoid(self.W_psi(h_delta_h_c))

                memory = torch.tanh(mu + torch.mul(f_psi, self.memory[:input_pts.shape[0],:]))
                new_memory = self.memory.clone()
                new_memory[:memory.shape[0],:] = memory
                self.memory = new_memory.detach()

                del new_memory      
                torch.cuda.empty_cache()    


            else:
                f_psi = torch.sigmoid(self.W_psi(h_delta_h_c))
                memory = torch.tanh(mu + torch.mul(f_psi, self.memory))
                self.memory = memory.detach()    

        # block 2
        sigma = self.M_line_delta(torch.cat((torch.mul(f_delta, memory), input_pts), dim=1)) # h: [batch_size, hidden_dim // 2]
        c =  self.M_line_c(torch.cat((torch.mul(f_c, memory), input_views), dim=1))  # h: [batch_size, hidden_dim // 2]


        del memory
        torch.cuda.empty_cache()
        outputs = torch.cat([c, sigma], -1)
        return outputs