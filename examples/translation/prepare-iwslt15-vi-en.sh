#!/bin/bash

set -e  # Dừng khi gặp lỗi

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

src=vi
tgt=en
lang=vi-en
prep=iwslt15.tokenized.$lang
orig=orig/$lang
tmp=$prep/tmp

mkdir -p $orig $tmp $prep

# Clone nếu thiếu
if [ ! -d "$SCRIPTS" ]; then
    echo "Cloning Moses github repository..."
    git clone https://github.com/moses-smt/mosesdecoder.git
fi

if [ ! -d "$BPEROOT" ]; then
    echo "Cloning Subword NMT repository..."
    git clone https://github.com/rsennrich/subword-nmt.git
fi

# Tải và giải nén
echo "Downloading data..."
cd $orig
curl -L -O https://github.com/stefan-it/nmt-en-vi/raw/master/data/train-en-vi.tgz
curl -L -O https://github.com/stefan-it/nmt-en-vi/raw/master/data/dev-2012-en-vi.tgz
curl -L -O https://github.com/stefan-it/nmt-en-vi/raw/master/data/test-2013-en-vi.tgz
tar -xvf train-en-vi.tgz
tar -xvf dev-2012-en-vi.tgz
tar -xvf test-2013-en-vi.tgz
cd ../../

echo "Pre-processing train data..."
for l in $src $tgt; do
    f=train.$l
    tok=train.tok.$l

    cat $orig/$f | \
        perl $REM_NON_PRINT_CHAR | \
        perl $NORM_PUNC -l $l > $tmp/tmp.norm.$l

    if [ "$l" = "vi" ]; then
        echo "Tokenizing Vietnamese using underthesea..."
        python tokenize_vi.py $tmp/tmp.norm.$l > $tmp/$tok
    else
        perl $TOKENIZER -threads 8 -l $l < $tmp/tmp.norm.$l > $tmp/$tok
    fi
done

echo "Cleaning train data..."
perl $CLEAN -ratio 1.5 $tmp/train.tok $src $tgt $tmp/train.clean 1 175

echo "Pre-processing valid/test data..."
for l in $src $tgt; do
    for prefix in tst2012 tst2013; do
        cat $orig/$prefix.$l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $NORM_PUNC -l $l > $tmp/tmp.norm.$prefix.$l

        if [ "$l" = "vi" ]; then
            python tokenize_vi.py $tmp/tmp.norm.$prefix.$l > $tmp/$prefix.tok.$l
        else
            perl $TOKENIZER -threads 8 -l $l < $tmp/tmp.norm.$prefix.$l > $tmp/$prefix.tok.$l
        fi
    done
done

echo "Preparing valid/test splits..."
cp $tmp/tst2012.tok.$src $tmp/valid.$src
cp $tmp/tst2012.tok.$tgt $tmp/valid.$tgt
cp $tmp/tst2013.tok.$src $tmp/test.$src
cp $tmp/tst2013.tok.$tgt $tmp/test.$tgt

# Gộp dữ liệu để học BPE
TRAIN=$tmp/train.clean.all
BPE_CODE=$prep/code
rm -f $TRAIN
cat $tmp/train.clean.$src $tmp/train.clean.$tgt > $TRAIN

echo "Learning BPE..."
python - <<EOF
from subword_nmt import learn_bpe
with open("$TRAIN", encoding="utf-8") as inp, open("$BPE_CODE", "w", encoding="utf-8") as out:
    learn_bpe.learn_bpe(inp, out, num_symbols=$BPE_TOKENS)
EOF

echo "Applying BPE..."
for L in $src $tgt; do
    for f in train.clean valid test; do
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f.$L > $prep/$f.$L
    done
done

echo "Data preprocessing complete."
