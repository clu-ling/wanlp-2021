### Dependencies
```bash
# see https://github.com/ipython/ipython/issues/12745
pip install -U jedi==0.17.2 parso==0.7.1
```

```python
gensim.models.Word2Vec.load('models/full_grams_cbow_100_twitter.mdl')
```

### Data
- [AraVec w2v embeddings](https://github.com/bakrianoo/aravec) (skip-gram using twitter data): 
  - https://bakrianoo.s3-us-west-2.amazonaws.com/aravec/full_uni_sg_300_twitter.zip

### Running the script
```bash
mkdir -p data/{out,w2v}
# NOTE: downloading this file may take awhile ...
wget https://bakrianoo.s3-us-west-2.amazonaws.com/aravec/full_uni_sg_300_twitter.zip | unzip full_uni_sg_300_twitter.zip | mv full_uni_sg_300_twitter.mdl* data/w2v/
rm -rf full_uni_sg_300_twitter.zip

# assuming the data is under data/
python classify_dialects.py \
--input "data/DA_train_labeled.tsv" \
--embeddings "data/w2v/full_uni_sg_300_twitter.mdl" \
--test "data/DA_dev_labeled.tsv" \
--out "data/out/dev-predictions.txt" \
--use-neg \
--vocab-size 150000 \
--max-epochs 20 \
--batch-size 64 \
--verbose
```