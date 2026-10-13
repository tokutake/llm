import argparse
import json
import pickle
import random

def main():
    ap=argparse.ArgumentParser(description='Generate text from a locally pretrained tiny GPT.')
    ap.add_argument('--prompt',default='The ')
    ap.add_argument('--tokens',type=int,default=200)
    ap.add_argument('--temperature',type=float,default=0.9)
    ap.add_argument('--checkpoint',default='checkpoint.bin')
    args=ap.parse_args()
    with open('model.json',encoding='utf-8') as f: meta=json.load(f)
    with open(args.checkpoint,'rb') as f: model=pickle.load(f)
    stoi={ch:i for i,ch in enumerate(meta['chars'])}
    unk=stoi.get(' ',0)
    ids=[stoi.get(ch,unk) for ch in args.prompt]
    random.seed()
    print(''.join(meta['chars'][i] for i in model.generate(ids,args.tokens,args.temperature)))

if __name__=='__main__': main()
