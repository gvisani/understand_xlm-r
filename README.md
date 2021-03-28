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

## Test inference of XLM-R pre-trained model

Run `python xlm_roberta.py`

