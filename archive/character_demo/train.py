import argparse
import json
import math
import pickle
import random
import time
from model import GPT


def main():
    ap = argparse.ArgumentParser(description='Train a tiny character GPT from random initialization.')
    ap.add_argument('--data', default='data.txt')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--context', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.003)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default='checkpoint.bin')
    args = ap.parse_args()
    random.seed(args.seed)
    with open(args.data, encoding='utf-8') as f: text = f.read()
    chars = sorted(set(text))
    if len(text) < 4: raise SystemExit('Corpus needs at least four characters.')
    stoi = {ch:i for i,ch in enumerate(chars)}
    encoded = [stoi[ch] for ch in text]
    if len(encoded) < 2: raise SystemExit('Corpus is too short.')
    model = GPT(len(chars), args.dim, args.heads, args.layers, args.context)
    start = time.time(); losses=[]
    for step in range(1,args.steps+1):
        end = random.randint(1,len(encoded)-1)
        begin = max(0,end-min(args.context,len(encoded)-1))
        seq = encoded[begin:end+1]
        loss = model.train_example(seq,args.lr)
        losses.append(loss)
        if step % 10 == 0 or step == 1:
            avg=sum(losses[-10:])/len(losses[-10:])
            print(f'step {step:4d}/{args.steps}  loss {avg:.3f}  elapsed {time.time()-start:.1f}s',flush=True)
    with open(args.out,'wb') as f: pickle.dump(model,f)
    with open('model.json','w',encoding='utf-8') as f:
        json.dump({'chars':chars,'dim':args.dim,'layers':args.layers,'heads':args.heads,'context':args.context},f,ensure_ascii=False,indent=2)
    print(f'Saved {args.out} and model.json')

if __name__ == '__main__': main()
