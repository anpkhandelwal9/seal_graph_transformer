import torch

class GAE(nn.Module):
    def __init__(self, feat_dim, embed_dim=16, hidden_dim=32):
        super(GAE, self).__init__()

        #self.lin = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.ReLU())
        self.c1 = GCNConv(feat_dim, hidden_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(hidden_dim, embed_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.25)
        self.de = DropEdge(0.8)

    def forward(self, x, ei, ew=None):
        ei = self.de(ei)
        x = self.c1(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)

        return self.c2(x, ei, edge_weight=ew)


class Recurrent(nn.Module):
    def __init__(self, feat_dim, out_dim=16, hidden_dim=32, hidden_units=1, lstm=False):
        super(Recurrent, self).__init__()

        self.gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=hidden_units
        ) if not lstm else \
            nn.LSTM(
                feat_dim, hidden_dim, num_layers=hidden_units
            )

        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(hidden_dim, out_dim)
        
        self.out_dim = out_dim 

    '''
    Expects (t, batch, feats) input
    Returns (t, batch, embed) embeddings of nodes at timesteps 0-t
    '''
    def forward(self, xs, h_0, include_h=False):
        xs = self.drop(xs)
        if type(h_0) != type(None):
            xs, h = self.gru(xs, h_0)
        else:
            xs, h = self.gru(xs)
        
        xs = self.drop(xs)
        if not include_h:
            return self.lin(xs)
        else:
            return self.lin(xs), h

'''

'''
def encode(self, xs, eis, mask_fn, ew_fn=None, start_idx=0):
    embeds=[]

    for i in range(len(eis)) :
        ei = mask_fn(start_idx + i)
        ew = ew_fn(start_idx + i) if ew_fn else None
        x = xs[start_idx + i] if self.dynamic_feats else xs
        z = self.gcn(x, ei, ew)
        embeds.append(z)

    return torch.stack(embeds)

'''
Inner product given edge list and embeddings at time t
'''

def decode(self, src, dst, z, as_probs=False):

    mul = self.drop(z[src]) * self.drop(z[dst])

    if self.use_predictor:
        return self.predictor(mul)

    dot = mul.sum(dim=1)
    logits = self.sig(dot)

    if as_probs:
        return self.__logits_to_probs(logits)
    
    return logits

'''
Given confidence scores of true samples and false samples,
return negative log likelihood 
'''
def calc_loss(self, t_scores, f_scores):
    EPS = 1e-6
    pos_loss = -torch.log(t_scores + EPS).mean()
    neg_loss = -torch.log(1 - f_scores + EPS).mean()

    return (1 - self.neg_weight) * pos_loss + self.neg_weight * neg_loss


'''
Expects a list of true edges and false edges from each time step.
Note: edge lists need not be the same length. Requires less preprocessing but doesn't utilize GPU/tensor ops as effectively as the batched fn  
'''
def loss_fn(self, ts, fs, zs):
    tot_loss = torch.zeros((1))
    T = len(ts)

    for i in range(T):
        t_src, t_dst = ts[i]
        f_src, f_dst = fs[i]
        z = zs[i]

        if self.dense_loss:
            tot_loss += full_adj_nll(ts[i], z)
        else:
            tot_loss += self.calc_loss(
                    self.decode(t_src, t_dst, z),
                    self.decode(f_src, f_dst, z)
                )
                
        return tot_loss.true_divide(T)


'''
Get scores for true/false embeddings to find ROC/AP scores.
Essentially the same as loss_fn but with no NLL 
Returns logits unless as_probs is True
'''
def score_fn(self, ts, fs, zs, as_probs=False):
    tscores = []
    fscores = []

    T = len(ts)

    for i in range(T):
        t_src, t_dst = ts[i]
        f_src, f_dst = fs[i]
        z = zs[i]

        tscores.append(self.decode(t_src, t_dst, z))
        fscores.append(self.decode(f_src, f_dst, z))

    tscores = torch.cat(tscores, dim=0)
    fscores = torch.cat(fscores, dim=0)

    if as_probs:
        tscores=self.__logits_to_probs(tscores)
        fscores=self.__logits_to_probs(fscores)

    return tscores, fscores


'''
Converts from log odds (what the encode method outputs) to probabilities
'''
def __logits_to_probs(self, logits):
    odds = torch.exp(logits)
    probs = odds.true_divide(1+odds)
    return probs