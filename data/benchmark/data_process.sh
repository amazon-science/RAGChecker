# !/bin/bash

RAWDATA_DIR="raw_data"
PROCESSED_DIR="processed_data"

if [ ! -d "$RAWDATA_DIR" ]; then
  mkdir $RAWDATA_DIR
fi
if [ ! -d "$PROCESSED_DIR" ]; then
  mkdir $PROCESSED_DIR
fi

# Process ClapNQ
if [ ! -d "$PROCESSED_DIR/clapnq" ]; then
  mkdir $PROCESSED_DIR/clapnq
fi
cd $RAWDATA_DIR
git clone https://github.com/primeqa/clapnq.git
cd ..
python process_clapnq.py

# Process NovelQA
if [ ! -d "$PROCESSED_DIR/novelqa" ]; then
  mkdir $PROCESSED_DIR/novelqa
fi
cd $RAWDATA_DIR
git clone https://huggingface.co/datasets/NovelQA/NovelQA
cd ..
python process_novelqa.py


# Process KIWI
if [ ! -d "$PROCESSED_DIR/kiwi" ]; then
  mkdir $PROCESSED_DIR/kiwi
fi
python process_kiwi.py


# Process BioASQ
if [ ! -d "$PROCESSED_DIR/bioasq" ]; then
  mkdir $PROCESSED_DIR/bioasq
fi
python process_bioasq.py


# Process FiQA
if [ ! -d "$PROCESSED_DIR/fiqa" ]; then
  mkdir $PROCESSED_DIR/fiqa
fi
python process_fiqa.py


# Process LoTTE
for domain in lifestyle science writing recreation technology
do
  if [ ! -d "$PROCESSED_DIR/$domain" ]; then
    mkdir $PROCESSED_DIR/$domain
  fi
done
python process_lotte.py