Download ollama version 0.18.1

curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.18.1 sh

ollama pull nomic-embed-text-v2-moe:latest

ollama pull snowflake-arctic-embed2:568m

Create a python Env: 

python3.12 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

The Dataset Stucture Expected in the following format

SOTABV2forSemTab2023/
├── CTA-SCH-R1/
│   ├── gt/
│   │   ├── supplementary_files/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── validation.csv
│   ├── Round1-SOTAB-CTA-SCH-Tables/
│   └── supplementary_files/
├── CTA-SCH-R2/
│   └── ...
└── CTA-SCH-R3/
    └── ...

How to run
functions.py has a function get_config which is the configuration file
each line has a comment explaining its purpose 
In the first run all fields should be False, in order to generate all the neccessary files (PREPROCESS_EMB_FILES_EXISTS,MODEL_EXISTS,PROTPYPES_EMB_EXISTS,CELL_DICT_EXISTS)
if some files already exists you can change the line to True in order to save time.

python -u cta_full_pipeline.py 2>&1 | tee pipeline.log

all logs are going to be redirected to pipeline.log