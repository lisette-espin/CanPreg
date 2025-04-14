# CanPreg
Cannabis use During Pregnancy: A Spatio-Temporal Analysis on Twitter

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

## Data
Aggregate values for survey & Twitter data, and country shape files: [https://doi.org/10.17605/OSF.IO/P5D9Y](https://doi.org/10.17605/OSF.IO/P5D9Y)
- Note that tweet and author ids are anonymized, and tweet text has been replaced by the CanPreg keywords contained in the tweet.

## How to cite

If you use any implementation from this repository, please cite both the repository and the corresponding paper as follows:

### Citing the GitHub repository

<pre>
  @software{canpreg2025code,
            author = {{Lisette Espín-Noboa}},
            title = {{CanPreg}},
            year = {2025},
            publisher = {GitHub},
            journal = {GitHub repository},
            howpublished = {\url{[https://github.com/lisette-espin/CanPreg](https://github.com/lisette-espin/CanPreg)}}}
</pre>

### Citing the ICWSM paper

<pre>
  @inproceedings{canpreg2025,
                 title={{Cannabis Use During Pregnancy: Insights from Online Discourse and Socioeconomic Indicators Across the USA and Canada}},
                 author={Espín-Noboa, Lisette and Farsiu, Nikou and Corsi, Daniel J. and Karsai Márton},
                 booktitle={{Proceedings of the international AAAI Conference on Web and Social Media}},
                 year={2025}}
</pre>

### Citing the dataset

<pre>
  @dataset{canpreg2025data,
           title={{Cannabis Use During Pregnancy: Insights from Online Discourse and Socioeconomic Indicators Across the USA and Canada}},
           author={Espín-Noboa, Lisette and Farsiu, Nikou},
           year={2025}
           url={\url{[https://doi.org/10.17605/OSF.IO/P5D9Y](https://doi.org/10.17605/OSF.IO/P5D9Y)}}}
</pre>

Thank you for citing our work! 🚀


