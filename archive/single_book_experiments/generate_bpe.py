import argparse, json, random
import numpy as np
from tinygrad import Tensor
from train_bpe import GPT
from tokenizer import BPETokenizer

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--prompt',default='The happiness of your life depends upon')
    ap.add_argument('--tokens',type=int,default=100); ap.add_argument('--temperature',type=float,default=.8)
    ap.add_argument('--checkpoint',default='meditations_bpe.bin'); args=ap.parse_args()
    cfg=json.load(open('meditations_bpe_config.json',encoding='utf-8'))
    tok=BPETokenizer.load('meditations_tokenizer.json')
    model=GPT(cfg['vocab'],cfg['dim'],cfg['heads'],cfg['layers'],cfg['context'])
    with np.load(args.checkpoint) as data:
        for i,p in enumerate(model.params): p.assign(Tensor(data[f'p{i}']))
    ids=tok.encode(args.prompt); random.seed()
    for _ in range(args.tokens):
        x=Tensor(np.asarray([ids[-cfg['context']:]],dtype=np.int32),dtype='int')
        logits=model(x)[0,-1,:]/max(args.temperature,1e-3)
        probs=logits.softmax(axis=-1).numpy().astype(np.float64)
        # Top-k sampling keeps a tiny model from spending probability on implausible tails.
        k=min(30,len(probs)); candidates=np.argpartition(probs,-k)[-k:]
        weights=probs[candidates]; weights/=weights.sum()
        ids.append(int(np.random.choice(candidates,p=weights)))
    print(tok.decode(ids))
if __name__=='__main__': main()
