"""Small BPE GPT pretraining with CPU and Apple Metal (MPS) support."""
import argparse, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from tokenizer import BPETokenizer

def choose_device(name):
    if name == 'auto':
        return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(name)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit('MPS was requested but is unavailable in this PyTorch/macOS environment.')
    return device

def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()

class Block(nn.Module):
    def __init__(self,dim,heads):
        super().__init__(); self.heads=heads; self.hd=dim//heads
        self.n1=nn.LayerNorm(dim); self.qkv=nn.Linear(dim,3*dim); self.proj=nn.Linear(dim,dim)
        self.n2=nn.LayerNorm(dim); self.ff=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim))
    def forward(self,x):
        b,t,d=x.shape; z=self.n1(x); q,k,v=self.qkv(z).chunk(3,dim=-1)
        def split(a): return a.view(b,t,self.heads,self.hd).transpose(1,2)
        a=nn.functional.scaled_dot_product_attention(split(q),split(k),split(v),is_causal=True)
        x=x+self.proj(a.transpose(1,2).contiguous().view(b,t,d))
        return x+self.ff(self.n2(x))

class GPT(nn.Module):
    def __init__(self,vocab,dim=96,heads=4,layers=2,context=128):
        super().__init__(); self.context=context
        self.tok=nn.Embedding(vocab,dim); self.pos=nn.Embedding(context,dim)
        self.blocks=nn.ModuleList([Block(dim,heads) for _ in range(layers)])
        self.norm=nn.LayerNorm(dim); self.out=nn.Linear(dim,vocab,bias=False)
        self.out.weight=self.tok.weight
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m,nn.Linear): nn.init.normal_(m.weight,std=.02); nn.init.zeros_(m.bias) if m.bias is not None else None
        elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,std=.02)
    def forward(self,ids):
        b,t=ids.shape; pos=torch.arange(t,device=ids.device)
        x=self.tok(ids)+self.pos(pos)[None,:,:]
        for block in self.blocks: x=block(x)
        return self.out(self.norm(x))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='philosophy_corpus.txt')
    ap.add_argument('--tokenizer',default='philosophy_tokenizer.json'); ap.add_argument('--steps',type=int,default=5000)
    ap.add_argument('--dim',type=int,default=96); ap.add_argument('--layers',type=int,default=2)
    ap.add_argument('--heads',type=int,default=4); ap.add_argument('--context',type=int,default=96)
    ap.add_argument('--batch',type=int,default=16); ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--seed',type=int,default=11); ap.add_argument('--out',default='philosophy_torch.pt')
    ap.add_argument('--save-every',type=int,default=1000); ap.add_argument('--init',default=None)
    ap.add_argument('--device',choices=['auto','cpu','mps'],default='auto',help='auto prefers MPS when available')
    args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    device=choose_device(args.device)
    tok=BPETokenizer.load(args.tokenizer); text=Path(args.data).read_text(encoding='utf-8')
    ids=torch.tensor(tok.encode(text),dtype=torch.long); vocab=len(tok.tokens)
    model=GPT(vocab,args.dim,args.heads,args.layers,args.context).to(device)
    if args.init:
        checkpoint=torch.load(args.init,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['state'])
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    print(f'{len(text):,} characters -> {len(ids):,} BPE tokens; vocab {vocab}; parameters {sum(p.numel() for p in model.parameters()):,}; device {device}',flush=True)
    sync(device); started=time.time(); losses=[]
    for step in range(1,args.steps+1):
        starts=torch.randint(0,len(ids)-args.context-1,(args.batch,))
        x=torch.stack([ids[s:s+args.context] for s in starts]).to(device)
        y=torch.stack([ids[s+1:s+args.context+1] for s in starts]).to(device)
        model.train(); logits=model(x); loss=nn.functional.cross_entropy(logits.reshape(-1,vocab),y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        losses.append(loss.item())
        if step==1 or step%100==0:
            sync(device)
            print(f'step {step:5d}/{args.steps} loss {sum(losses[-100:])/len(losses[-100:]):.3f} elapsed {time.time()-started:.1f}s',flush=True)
        if args.save_every and step%args.save_every==0:
            torch.save({'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':{'vocab':vocab,'dim':args.dim,'heads':args.heads,'layers':args.layers,'context':args.context}},args.out)
    sync(device)
    torch.save({'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':{'vocab':vocab,'dim':args.dim,'heads':args.heads,'layers':args.layers,'context':args.context}},args.out)
    print(f'Saved {args.out}; final mean loss {sum(losses[-100:])/len(losses[-100:]):.3f}',flush=True)
if __name__=='__main__': main()
