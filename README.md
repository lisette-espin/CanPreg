# CanPreg
Cannabis use During Pregnancy: A Spatio-Temporal Analysis on Twitter

## Data
Aggregate values for survey & Twitter data, and contry shape files: https://osf.io/p5d9y/?view_only=6b440b8309a840bf9b4f946c48bd83fc
- Note that tweet and author ids are anonymized, and tweet text has been replaced by the CanPreg keywords contained in the tweet.

```
data/  
├── annotations/                      *(annotation files for ground-truth sample)*
├── census/                           *(survey data)*  
├── geo_boundaries/                   *(shape files for USA & CAN)*
└── twitter_places/                   *(total counts per year and region)*  

results/  
├── census_twitter_data/              *(aggregate counts and norm counts per region and year)*
├── corpus_33K_users_valid_anon.csv   *(USA & CAN users metadata)*
├── corpus_53K_tweets_valid_anon.csv  *(USA & CAN tweets metadata)*
└── RQ3_corpus_embeddings_small.pt    *(SBERT model to perform fast clustering)*
```
