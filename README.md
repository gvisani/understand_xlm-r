# understand_xlm-r

## Create Conda environment

Run the following commands:  
`conda env create --file xlm-r.yml`  
`conda activate xlm-r`

For pytorch with cpu only, run:  
`pip install transformers[torch]`

Otherwise, for gpu support, first intall pytorch with desired cuda verson, then run:
`pip install transformers`


## Download Universal Dependencies datasets

1. Download zip file from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3424. Unzip into folder called *UD2.7*.
2. Run `unzip_ud.py`



## Scripts for data preparation

The script `xlm_roberta.py` takes a PUD dataset file (or a file with the same format) and created a data structure containing hidden states and attention weights for all tokens in all sentences in the file.
This script generates a pkl file that is used as input for the POS classifiers (`diagnostic_classifiers_pos.py`, `pos_statistics.py`) and for generating attention weights heatmaps (`gen_hetamap.py`).

The script `xlm_roberta_multilingual_dicts.py` takes a file with bilingual lexicons (two columns of parallel words in two languages, one word per line, tab-separated, top row is the name of the language) and creates a data structure, saved as a pkl file, containing hidden states and attention weights for each token.

### POS experiment


