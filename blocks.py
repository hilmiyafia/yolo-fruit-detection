
import torch

class Conv(torch.nn.Module):

    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        c_in = round(c_in)
        c_out = round(c_out)
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            torch.nn.BatchNorm2d(c_out),
            torch.nn.SiLU())
    
    def forward(self, x):
        return self.layers(x)
    
class Bottleneck(torch.nn.Module):

    def __init__(self, c):
        super().__init__()
        self.layers = torch.nn.Sequential(Conv(c, c), Conv(c, c))
        
    def forward(self, x):
        return self.layers(x) + x
     
class C2f(torch.nn.Module):
    
    def __init__(self, c_in, c_out, n=3):
        super().__init__()
        c_in = round(c_in)
        c_out = round(c_out)
        n = round(n)
        d = c_out // 2
        self.conv1 = Conv(c_in, c_out, 1, 1, 0)
        self.layers = torch.nn.ModuleList([Bottleneck(d) for _ in range(n)])
        self.conv2 = Conv(d * (n + 2), c_out, 1, 1, 0)
    
    def forward(self, x):
        z = self.conv1(x)
        z = [z, z.chunk(2, 1)[1]]
        for layer in self.layers: 
            z.append(layer(z[-1]))
        return self.conv2(torch.cat(z[0:1] + z[2:], 1))
