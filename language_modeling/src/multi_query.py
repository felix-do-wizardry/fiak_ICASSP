import torch
from torch import nn

class MultiQueryAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False):
        super(MultiQueryAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        assert not carry_over_fast_weight, "Not supported."
        # multihead attention
        # [hlen x bsz x n_head x d_head]
        # print(h.shape)
        # assert 1==2
        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.d_head)

        # import pdb; pdb.set_trace()

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(
                    attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head]
        # -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbd->ibnd', (attn_prob, head_v))
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



# Standard multihead attention.
class MGKMultiQueryAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False, update_mode='rbf2keys'):
        super(MGKMultiQueryAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.update_mode = update_mode

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        if self.update_mode == 'hard2keys' or self.update_mode == 'soft2keys' or self.update_mode == 'rbf2keys':
            self.kv_net = nn.Linear(d_model, 3 * d_head, bias=False)
        else:
            self.kv_net = nn.Linear(d_model, 2 * d_head, bias=False)
            
        # for mgk
        if self.update_mode == 'hard' or self.update_mode == 'soft' or self.update_mode == 'rbf':
            self.mu = nn.Parameter((torch.empty(2, n_head, d_head).normal_(mean = 0.0, std = .5))*torch.tensor([0.,1.])[:, None, None], requires_grad= True)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        
        if self.update_mode == 'soft' or self.update_mode == 'soft2keys':
            self.register_buffer("pi", 0.5 * torch.ones(self.n_head, 1, 1, 256, requires_grad= False))
            # self.register_buffer("pi", 0.5 * torch.ones(self.n_head, 1, 1, 1, requires_grad= False))
        
        if self.update_mode == 'rbf' or self.update_mode == 'rbf2keys':

            ## for model 256
            self.pi0 = nn.Parameter(0.5 * torch.ones(self.n_head, 1, 1, 256), requires_grad= True)
            self.pi1 = nn.Parameter(0.5 * torch.ones(self.n_head, 1, 1, 256), requires_grad= True)

            ###for model 384
            # self.pi0 = nn.Parameter(0.5 * torch.ones(self.n_head, 1, 1, 384), requires_grad= True)
            # self.pi1 = nn.Parameter(0.5 * torch.ones(self.n_head, 1, 1, 384), requires_grad= True)
        
    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        assert not carry_over_fast_weight, "Not supported."
        # multihead attention
        # [hlen x bsz x n_head x d_head]
        # print(h.shape)
        # assert 1==2
        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)
        
        head_q = self.q_net(h)
        if self.update_mode == 'hard2keys' or self.update_mode == 'soft2keys' or self.update_mode == 'rbf2keys':
            head_k, head_k1, head_v = torch.chunk(self.kv_net(c), 3, -1)
        else:
            head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        # print(head_k1.shape)
        # assert 1==2

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1),1, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.d_head)
        if self.update_mode == 'hard2keys' or self.update_mode == 'soft2keys' or self.update_mode == 'rbf2keys':
            head_k1 = head_k1.view(c.size(0), c.size(1), 1, self.d_head)
        
        if self.update_mode == 'hard2keys' or self.update_mode == 'soft2keys' or self.update_mode == 'rbf2keys':
            QK_distance0 = (-self.scale/2)*torch.square(torch.cdist(head_q.transpose(0,2), head_k.transpose(0,2))) 
            QK_distance1 = (-3*self.scale/2)*torch.square(torch.cdist(head_q.transpose(0,2), head_k1.transpose(0,2))) 
            # QK_distance1 = (-self.scale/2)*torch.square(torch.cdist(head_q.transpose(0,2), head_k1.transpose(0,2))) 
        else:
            QK_distance0 = (-self.scale/2)*torch.square(torch.cdist(head_q.transpose(0,2), (head_k - self.mu[0]).transpose(0,2))) # n_head x bsz x qlen x klen
            QK_distance1 = (-self.scale/2)*torch.square(torch.cdist(head_q.transpose(0,2), (head_k - self.mu[1]).transpose(0,2))) # n_head x bsz x qlen x klen
        
        if self.update_mode == 'hard' or self.update_mode == 'hard2keys':
            attn_score = torch.maximum(QK_distance0, QK_distance1)
            attn_score = attn_score.permute(2, 3, 1, 0)

            # [qlen x klen x bsz x n_head]
            if attn_mask is not None and attn_mask.any().item():
                if attn_mask.dim() == 2:
                    attn_score.masked_fill_(
                        attn_mask[None,:,:,None], -float('inf'))
                elif attn_mask.dim() == 3:
                    attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

            # [qlen x klen x bsz x n_head]
            attn_prob = F.softmax(attn_score, dim=1)
            
        elif self.update_mode == 'soft' or self.update_mode == 'soft2keys':
            pi = self.pi.clone().detach()
            attn_prob = pi[:,:,:,:c.size(0)] * torch.exp(QK_distance0) + (1.0 - pi[:,:,:,:c.size(0)]) * torch.exp(QK_distance1)
            # attn_prob = pi * torch.exp(QK_distance0) + (1.0 - pi) * torch.exp(QK_distance1)
            if self.training is True:
                resp0 = pi[:,:,:,:c.size(0)] * torch.exp(QK_distance0) / (attn_prob + 1e-6)
                # resp0 = pi * torch.exp(QK_distance0) / (attn_prob + 1e-6)
                pi_new = torch.sum(resp0, dim=(1,2), keepdim=True)/(h.size(0) * h.size(1))
                pi_new = torch.cat((pi_new, pi[:,:,:,c.size(0):]), dim=3)
                # pi_new = torch.sum(resp0, dim=(1,2,3), keepdim=True)/(h.size(0) * h.size(1) * c.size(0))
                pi_new = pi_new.to(h)
                self.pi.copy_(pi_new.detach())
            
            attn_prob = attn_prob.permute(2, 3, 1, 0)
            
            if attn_mask is not None and attn_mask.any().item():
                if attn_mask.dim() == 2:
                    attn_prob.masked_fill_(
                        attn_mask[None,:,:,None], 0.0)
                elif attn_mask.dim() == 3:
                    attn_prob.masked_fill_(attn_mask[:,:,:,None], 0.0)
            
            attn_prob = attn_prob / ((attn_prob.sum(dim=1))[:, None, :, :] + 1e-6)
            
        else:
            attn_prob = self.pi0[:,:,:,:c.size(0)] * torch.exp(QK_distance0) + self.pi1[:,:,:,:c.size(0)] * torch.exp(QK_distance1)
            # attn_prob = self.pi0 * torch.exp(QK_distance0) + self.pi1 * torch.exp(QK_distance1)
            attn_prob = attn_prob.permute(2, 3, 1, 0)
            
            if attn_mask is not None and attn_mask.any().item():
                if attn_mask.dim() == 2:
                    attn_prob.masked_fill_(
                        attn_mask[None,:,:,None], 0.0)
                elif attn_mask.dim() == 3:
                    attn_prob.masked_fill_(attn_mask[:,:,:,None], 0.0)
                    
            attn_prob = attn_prob / ((attn_prob.sum(dim=1))[:, None, :, :] + 1e-6)
          
                    
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head]
        # -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbd->ibnd', (attn_prob, head_v))
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