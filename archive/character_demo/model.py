"""A compact character-level GPT implemented with Python's standard library."""
import math
import random

def randmat(r,c,s=.02): return [[random.gauss(0,s) for _ in range(c)] for _ in range(r)]
def mv(x,w): return [sum(a*b for a,b in zip(row,x)) for row in w]
def softmax(x):
    m=max(x); e=[math.exp(v-m) for v in x]; z=sum(e); return [v/z for v in e]

class GPT:
    def __init__(self,vocab,dim=128,heads=4,layers=2,context=128):
        assert dim%heads==0
        self.vocab,self.dim,self.heads,self.layers,self.context=vocab,dim,heads,layers,context
        self.hd=dim//heads; self.tok=randmat(vocab,dim); self.pos=randmat(context,dim); self.blocks=[]
        for _ in range(layers):
            self.blocks.append({'q':randmat(dim,dim),'k':randmat(dim,dim),'v':randmat(dim,dim),
                'o':randmat(dim,dim),'f1':randmat(4*dim,dim),'f2':randmat(dim,4*dim)})
        self.out=randmat(vocab,dim)

    def forward(self,ids):
        ids=ids[-self.context:]; n=len(ids)
        x=[[self.tok[t][d]+self.pos[i][d] for d in range(self.dim)] for i,t in enumerate(ids)]
        cache=[]
        for b in self.blocks:
            residual=x; q=[mv(r,b['q']) for r in x]; k=[mv(r,b['k']) for r in x]; v=[mv(r,b['v']) for r in x]
            att=[[0.]*self.dim for _ in range(n)]; probs_by_head=[]
            for h in range(self.heads):
                s,e=h*self.hd,(h+1)*self.hd; head=[]
                for i in range(n):
                    p=softmax([sum(q[i][d]*k[j][d] for d in range(s,e))/math.sqrt(self.hd) for j in range(i+1)])
                    head.append(p)
                    for j,pj in enumerate(p):
                        for d in range(s,e): att[i][d]+=pj*v[j][d]
                probs_by_head.append(head)
            projected=[mv(r,b['o']) for r in att]
            x1=[[a+c for a,c in zip(r,p)] for r,p in zip(residual,projected)]
            pre=[mv(r,b['f1']) for r in x1]; hidden=[[max(0.,a) for a in r] for r in pre]
            ff=[mv(r,b['f2']) for r in hidden]; x=[[a+c for a,c in zip(r,p)] for r,p in zip(x1,ff)]
            cache.append((q,k,v,att,probs_by_head,x1,pre,hidden))
        return softmax(mv(x[-1],self.out)),(ids,x,cache)

    def train_example(self,seq,lr):
        probs,(ids,x,caches)=self.forward(seq[:-1]); target=seq[-1]
        loss=-math.log(max(probs[target],1e-12)); g=probs[:]; g[target]-=1
        # Accumulate gradients first. No parameter changes until backpropagation is complete.
        grads=[]
        def outer(w,a,b):
            grads.append((w,[[gi*aj for aj in a] for gi in b]))
        outer(self.out,x[-1],g)
        dx=[sum(g[v]*self.out[v][j] for v in range(self.vocab)) for j in range(self.dim)]
        gtok={}; gpos=[0.]*self.dim
        for li in reversed(range(self.layers)):
            b=self.blocks[li]; q,k,v,att,pbh,x1,pre,hidden=caches[li]; last=len(ids)-1
            dh=[sum(dx[o]*b['f2'][o][j] for o in range(self.dim)) for j in range(4*self.dim)]
            for j in range(4*self.dim):
                if pre[last][j]<=0: dh[j]=0.
            outer(b['f2'],hidden[last],dx); outer(b['f1'],x1[last],dh)
            dx1=[dx[j]+sum(dh[h]*b['f1'][h][j] for h in range(4*self.dim)) for j in range(self.dim)]
            datt=mv(dx1,b['o']); outer(b['o'],att[last],dx1)
            dq=[0.]*self.dim; dk=[[0.]*self.dim for _ in ids]; dv=[[0.]*self.dim for _ in ids]
            for h in range(self.heads):
                s,e=h*self.hd,(h+1)*self.hd; ps=pbh[h][last]
                dp=[sum(datt[d]*v[j][d] for d in range(s,e)) for j in range(last+1)]
                avg=sum(ps[j]*dp[j] for j in range(len(ps)))
                for j in range(last+1):
                    ds=ps[j]*(dp[j]-avg)
                    for d in range(s,e):
                        dv[j][d]+=ps[j]*datt[d]; dq[d]+=ds*k[j][d]/math.sqrt(self.hd); dk[j][d]+=ds*q[last][d]/math.sqrt(self.hd)
            outer(b['q'],x1[last],dq)
            for j in range(len(ids)):
                outer(b['k'],x1[j],dk[j]); outer(b['v'],x1[j],dv[j])
            dx=dx1[:]
            for j in range(len(ids)):
                backq=mv(dq,b['q']) if j==last else [0.]*self.dim
                backk=mv(dk[j],b['k']); backv=mv(dv[j],b['v'])
                dx=[dx[d]+backq[d]+backk[d]+backv[d] for d in range(self.dim)]
        for d,val in enumerate(dx):
            gtok.setdefault(ids[-1],[0.]*self.dim)[d]+=val; gpos[d]+=val
        grads.extend([(self.tok,[(gtok.get(i,[0.]*self.dim)) for i in range(len(self.tok))]),
                      (self.pos,[[0.]*self.dim for _ in self.pos])])
        grads[-1][1][len(ids)-1]=gpos
        for w,gw in grads:
            for i,row in enumerate(gw):
                for j,val in enumerate(row):
                    if val: w[i][j]-=lr*val
        return loss

    def generate(self,ids,count,temperature=.9):
        out=list(ids)
        for _ in range(count):
            p,_=self.forward(out[-self.context:]); p=softmax([math.log(max(v,1e-12))/max(temperature,1e-3) for v in p])
            r=random.random(); total=0.
            for i,v in enumerate(p):
                total+=v
                if total>=r: out.append(i); break
        return out
