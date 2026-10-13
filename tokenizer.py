"""A small, dependency-free word-aware BPE tokenizer for the training demo."""
import json
import re
from collections import Counter
from pathlib import Path

PRETOKEN_RE = re.compile(r"\s+|[^\W\d_]+(?:['’][^\W\d_]+)*|\d+|[^\w\s]", re.UNICODE)

class BPETokenizer:
    def __init__(self, alphabet, merges):
        self.alphabet = list(alphabet)
        self.merges = [tuple(x) for x in merges]
        self.tokens = self.alphabet[:]
        for a,b in self.merges:
            self.tokens.append(a+b)
        self.stoi = {t:i for i,t in enumerate(self.tokens)}

    @classmethod
    def train(cls, text, vocab_size=512):
        counts = Counter(PRETOKEN_RE.findall(text))
        alphabet = sorted(set(''.join(counts)))
        sequences = {word:list(word) for word in counts}
        merges=[]
        target=max(vocab_size,len(alphabet))
        while len(alphabet)+len(merges)<target:
            pairs=Counter()
            for word,seq in sequences.items():
                weight=counts[word]
                for pair in zip(seq,seq[1:]): pairs[pair]+=weight
            if not pairs: break
            pair,freq=max(pairs.items(),key=lambda item:(item[1],item[0]))
            merged=pair[0]+pair[1]
            if freq<2: break
            merges.append(pair)
            for word,seq in sequences.items():
                out=[]; i=0
                while i<len(seq):
                    if i+1<len(seq) and seq[i]==pair[0] and seq[i+1]==pair[1]:
                        out.append(merged); i+=2
                    else: out.append(seq[i]); i+=1
                sequences[word]=out
        return cls(alphabet,merges)

    def encode(self,text):
        ids=[]; cache={}
        for piece in PRETOKEN_RE.findall(text):
            cached=cache.get(piece)
            if cached is not None:
                ids.extend(cached); continue
            seq=list(piece)
            for a,b in self.merges:
                merged=a+b; out=[]; i=0
                while i<len(seq):
                    if i+1<len(seq) and seq[i]==a and seq[i+1]==b:
                        out.append(merged); i+=2
                    else: out.append(seq[i]); i+=1
                seq=out
            cached=tuple(self.stoi[t] for t in seq); cache[piece]=cached; ids.extend(cached)
        return ids

    def decode(self,ids):
        return ''.join(self.tokens[int(i)] for i in ids)

    def save(self,path):
        Path(path).write_text(json.dumps({'alphabet':self.alphabet,'merges':self.merges},ensure_ascii=False),encoding='utf-8')

    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(d['alphabet'],d['merges'])
