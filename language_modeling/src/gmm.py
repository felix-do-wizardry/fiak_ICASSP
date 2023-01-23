import torch
from torch import nn
class GMMAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False, update_mode='gd'):
        super(GMMAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.update_mode = update_mode

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        
        self.pi = nn.Parameter(torch.ones(256, 256, 1, self.n_head), requires_grad = True) # qlen x klen x 1 x n_head
        
    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        assert not carry_over_fast_weight, "Not supported."
        # multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)
        
        head_q = self.q_net(h) # shape: [256, 48, 128] = [hlen, bsz, d_feature]
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)
        
        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)
        
        QK_distance0 = (-self.scale/2.0)*torch.square(torch.cdist(head_q.transpose(0,2), head_k.transpose(0,2))) # n_head x bsz x qlen x klen
        QK_distance0 = QK_distance0.permute(2, 3, 1, 0) # qlen x klen x bsz x n_head
        # 
        # attn_prob = torch.clamp(torch.square(self.pi), dim = -1) * torch.exp(QK_distance0) # qlen x klen x bsz x n_head
        print('pi_mask', self.pi_mask)
        attn_prob = self.pi_mask * torch.exp(QK_distance0)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_prob.masked_fill_(attn_mask[None,:,:,None], 0.0)
            elif attn_mask.dim() == 3:
                attn_prob.masked_fill_(attn_mask[:,:,:,None], 0.0)
                    
        attn_prob = attn_prob / ((attn_prob.sum(dim=1))[:, None, :, :] + 1e-6)
        
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head]
        # -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)
        
        return output