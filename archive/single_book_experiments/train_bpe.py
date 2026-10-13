"""Train a small BPE-token GPT using tinygrad's automatic differentiation."""
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
from tinygrad import Tensor, Context
from tinygrad.nn.optim import Adam
from tokenizer import BPETokenizer

class GPT:
    def __init__(self,vocab,dim=64,heads=4,layers=2,context=96):
        self.vocab,self.dim,self.heads,self.layers,self.context=vocab,dim,heads,layers,context
        self.hd=dim//heads
        def param(shape,scale=.02):
            arr=np.random.randn(*shape).astype(np.float32)*scale
            return Tensor(arr)
        self.tok=param((vocab,dim)); self.pos=param((context,dim))
        self.blocks=[]
        for _ in range(layers):
            self.blocks.append({'q':param((dim,dim)),'k':param((dim,dim)),'v':param((dim,dim)),
                'o':param((dim,dim)),'f1':param((4*dim,dim)),'f2':param((dim,4*dim)),
                'n1g':param((dim,),1.0),'n1b':param((dim,),0.0),'n2g':param((dim,),1.0),'n2b':param((dim,),0.0)})
        self.ng=param((dim,),1.0); self.nb=param((dim,),0.0); self.out=param((vocab,dim))
        self.params=[self.tok,self.pos,self.out,self.ng,self.nb]
        for b in self.blocks: self.params.extend(b.values())
        mask=np.triu(np.ones((context,context),dtype=np.float32)*-1e9,k=1)
        self.mask=Tensor(mask.reshape(1,1,context,context))

    @staticmethod
    def norm(x,g,b):
        mean=x.mean(axis=-1,keepdim=True); var=((x-mean)*(x-mean)).mean(axis=-1,keepdim=True)
        return (x-mean)*(var+1e-5).rsqrt()*g+b

    def __call__(self,ids):
        B,T=ids.shape
        x=ids.one_hot(self.vocab) @ self.tok + self.pos[:T].reshape(1,T,self.dim)
        for b in self.blocks:
            z=self.norm(x,b['n1g'],b['n1b'])
            q=(z @ b['q']).reshape(B,T,self.heads,self.hd).transpose(1,2)
            k=(z @ b['k']).reshape(B,T,self.heads,self.hd).transpose(1,2)
            v=(z @ b['v']).reshape(B,T,self.heads,self.hd).transpose(1,2)
            scores=(q @ k.transpose(-1,-2))*(1/math.sqrt(self.hd))+self.mask[:,:,:T,:T]
            a=(scores.softmax(axis=-1) @ v).transpose(1,2).reshape(B,T,self.dim)
            x=x+a @ b['o']
            z=self.norm(x,b['n2g'],b['n2b'])
            x=x+(z @ b['f1'].transpose(0,1)).relu() @ b['f2'].transpose(0,1)
        return self.norm(x,self.ng,self.nb) @ self.out.transpose(0,1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='meditations_corpus.txt'); ap.add_argument('--steps',type=int,default=2000)
    ap.add_argument('--vocab',type=int,default=512); ap.add_argument('--dim',type=int,default=64)
    ap.add_argument('--layers',type=int,default=2); ap.add_argument('--heads',type=int,default=4)
    ap.add_argument('--context',type=int,default=96); ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--lr',type=float,default=.001); ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--tokenizer',default=None,help='Reuse a saved BPE tokenizer instead of retraining it')
    ap.add_argument('--init',default=None,help='Initialize weights from a previous checkpoint')
    ap.add_argument('--out',default='meditations_bpe.bin'); args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); Tensor.manual_seed(args.seed)
    text=Path(args.data).read_text(encoding='utf-8')
    if args.tokenizer:
        tok=BPETokenizer.load(args.tokenizer)
    else:
        print(f'Training BPE tokenizer on {len(text):,} characters...',flush=True)
        tok=BPETokenizer.train(text,args.vocab)
    ids=tok.encode(text)
    print(f'Corpus: {len(ids):,} tokens, vocabulary: {len(tok.tokens)}',flush=True)
    data=np.asarray(ids,dtype=np.int32)
    model=GPT(len(tok.tokens),args.dim,args.heads,args.layers,args.context)
    if args.init:
        with np.load(args.init) as saved:
            for i,p in enumerate(model.params): p.assign(Tensor(saved[f'p{i}']))
    opt=Adam(model.params,lr=args.lr); start=time.time(); running=[]
    with Context(TRAINING=1):
        for step in range(1,args.steps+1):
            starts=np.random.randint(0,len(data)-args.context-1,size=args.batch)
            xx=np.stack([data[s:s+args.context] for s in starts]); yy=np.stack([data[s+1:s+args.context+1] for s in starts])
            x=Tensor(xx,dtype='int'); y=Tensor(yy,dtype='int')
            logits=model(x); loss=logits.reshape(-1,len(tok.tokens)).sparse_categorical_crossentropy(y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            running.append(loss.item())
            if step==1 or step%50==0:
                avg=sum(running[-50:])/len(running[-50:]); print(f'step {step:4d}/{args.steps} loss {avg:.3f} elapsed {time.time()-start:.1f}s',flush=True)
    # Save plain parameter arrays to remain independent of tinygrad internals at inference.
    weights=[]
    def collect(t): weights.append(np.asarray(t.numpy()))
    for p in model.params: collect(p)
    with open(args.out,'wb') as f: np.savez(f,**{f'p{i}':w for i,w in enumerate(weights)})
    tok.save('meditations_tokenizer.json')
    Path('meditations_bpe_config.json').write_text(json.dumps({'vocab':len(tok.tokens),'dim':args.dim,'heads':args.heads,'layers':args.layers,'context':args.context,'param_shapes':[list(w.shape) for w in weights]},indent=2),encoding='utf-8')
    print(f'Saved {args.out}, tokenizer and config. Final loss {sum(running[-50:])/len(running[-50:]):.3f}')
if __name__=='__main__': main()
