import argparse
import torch
from train_torch import GPT
from train_torch import choose_device
from tokenizer import BPETokenizer

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--prompt',default='The happiness of your life depends upon')
    ap.add_argument('--tokens',type=int,default=120); ap.add_argument('--temperature',type=float,default=.7)
    ap.add_argument('--top-k',type=int,default=20); ap.add_argument('--checkpoint',default='philosophy_torch.pt')
    ap.add_argument('--tokenizer',default='philosophy_tokenizer.json')
    ap.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto',help='auto prefers CUDA, then MPS, then CPU')
    ap.add_argument('--seed',type=int,default=None,help='fixed seed for reproducible sampling')
    args=ap.parse_args(); torch.set_num_threads(6)
    device=choose_device(args.device)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False); cfg=checkpoint['config']
    tok=BPETokenizer.load(args.tokenizer); model=GPT(**cfg).to(device); model.load_state_dict(checkpoint['state']); model.eval()
    ids=tok.encode(args.prompt)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    else:
        torch.seed()
    with torch.no_grad():
        for _ in range(args.tokens):
            x=torch.tensor([ids[-cfg['context']:]],dtype=torch.long,device=device); logits=model(x)[0,-1]/max(args.temperature,.01)
            vals,ix=torch.topk(logits,min(args.top_k,len(logits))); p=torch.softmax(vals,dim=-1)
            ids.append(int(ix[torch.multinomial(p,1)].item()))
    print(tok.decode(ids))
if __name__=='__main__': main()
