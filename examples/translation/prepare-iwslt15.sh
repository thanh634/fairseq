#!/usr/bin/env bash
#
# Chuẩn bị dữ liệu IWSLT15 English-Vietnamese
# Tham khảo từ MIXER + fairseq

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

# IWSLT15 English-Vietnamese
URL="https://dl.fbaipublicfiles.com/fairseq/data/iwslt15_en_vi.tgz"
GZ=iwslt15_en_vi.tgz

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=vi
lang=en-vi
prep=iwslt15.tokenized.en-vi
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

echo "Downloading data from ${URL}..."
cd $orig
wget "$URL"

if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf $GZ
cd ..

echo "Pre-processing train data..."
for l in $src $tgt; do
    f=train.$lang.$l
    tok=train.tok.$lang.$l

    cat $orig/$f | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $tmp/$tok
done

perl $CLEAN -ratio 1.5 $tmp/train.tok.$lang $src $tgt $tmp/train.clean 1 175

for l in $src $tgt; do
    perl $LC < $tmp/train.clean.$l > $tmp/train.$lang.$l
done

echo "Pre-processing test data..."
for l in $src $tgt; do
    for f in tst2012.$lang.$l tst2013.$lang.$l; do
        cat $orig/$f | \
        perl $TOKENIZER -threads 8 -l $l | \
        perl $LC > $tmp/$f
    done
done

echo "Creating train, valid, test splits..."
for l in $src $tgt; do
    awk '{if (NR%23 == 0)  print $0; }' $tmp/train.$lang.$l > $tmp/valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/train.$lang.$l > $tmp/train.$l

    cat $tmp/tst2012.$lang.$l $tmp/tst2013.$lang.$l > $tmp/test.$l
done

TRAIN=$tmp/train.en-vi
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "Learning BPE on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "Applying BPE to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done

echo "All done. Tokenized & BPE-processed data is saved in '$prep'"
