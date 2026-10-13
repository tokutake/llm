"""Download and clean Natsume Soseki's Aozora Bunko text for local training."""
import re
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from tokenizer import BPETokenizer

URL = 'https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip'
OUTPUT = Path('wagahai_corpus.txt')
TOKENIZER = Path('wagahai_tokenizer.json')

def main():
    with urllib.request.urlopen(URL, timeout=30) as response:
        archive = zipfile.ZipFile(BytesIO(response.read()))
    source_name = next(name for name in archive.namelist() if name.endswith('.txt'))
    source = archive.read(source_name).decode('shift_jis')

    # Start at the first chapter, excluding Aozora's notation guide and end
    # matter while keeping chapter headings and the story itself.
    start = source.find('［＃８字下げ］一')
    end = source.find('底本：', start)
    if start < 0 or end < 0:
        raise RuntimeError('Could not locate the expected Aozora text boundaries.')
    text = source[start:end]
    text = re.sub(r'［＃.*?］', '', text)  # Aozora editorial/layout annotations
    text = re.sub(r'｜?([^｜《\s]+)《[^》]*》', r'\1', text)  # ruby readings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t\u3000]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip() + '\n'
    OUTPUT.write_text(text, encoding='utf-8')
    print(f'Wrote {OUTPUT}: {len(text):,} characters from {source_name}')
    tokenizer = BPETokenizer.train(text, vocab_size=4096)
    tokenizer.save(TOKENIZER)
    print(f'Wrote {TOKENIZER}: {len(tokenizer.tokens):,} tokens')

if __name__ == '__main__':
    main()
