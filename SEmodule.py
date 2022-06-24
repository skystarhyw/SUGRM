import torch
from torch import nn


class SELayer(nn.Module):
    def __init__(self, cl, ca, cv, reduction=4):
        super(SELayer, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.cz = (cl + ca + cv) // reduction
        self.wg = nn.Linear(cl + ca + cv, self.cz)
        self.wa = nn.Linear(self.cz, ca)
        self.wv = nn.Linear(self.cz, cv)
        self.wl = nn.Linear(self.cz, cl)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, l, a, v):
        sa = self.squeeze(a.permute(0,2,1))
        sv = self.squeeze(v.permute(0,2,1))
        sl = self.squeeze(l.permute(0,2,1))
        z = self.wg(torch.cat((sa, sv, sl), dim=1).permute(0,2,1))
        z = self.relu(z)
        ea = self.wa(z)
        ev = self.wv(z)
        el = self.wl(z)
        l_new = torch.mul(self.sigmoid(el), 2) * l
        v_new = torch.mul(self.sigmoid(ev), 2) * v
        a_new = torch.mul(self.sigmoid(ea), 2) * a
        return l_new, v_new, a_new


if __name__ == "__main__":
    a = torch.randn(48, 41, 74)     # (B, L, F)
    v = torch.randn(48, 42, 47)
    l = torch.randn(48, 40, 768)
    se_model = SELayer(l.shape[2], a.shape[2], v.shape[2])
    output = se_model(l, a, v)
    print(output)