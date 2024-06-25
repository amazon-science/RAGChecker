
# chunk ablation
for chunk_size in 100 600
do
python chunking.py --chunk_size $chunk_size \
    --data_names kiwi finance
done

for overlap in 0.0 0.4
do
python chunking.py --overlap_ratio $overlap \
    --data_names kiwi finance
done

