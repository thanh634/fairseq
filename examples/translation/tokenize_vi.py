import sys
from underthesea import word_tokenize
import io
import os

if len(sys.argv) < 2:
    print("Usage: python tokenize_vi.py <input_file>", file=sys.stderr)
    sys.exit(1)

input_file = sys.argv[1]

# Reconfigure stdout to use UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        print(word_tokenize(line.strip(), format="text"))
