# NLP702 Project Midterm Code
To run any of the dialectal generation code you need to install multi-value the instructions are include below (make sure to use python 3.6.9):
## Multi-VALUE: The VernAcular Language Understanding Evaluation benchmark 

### Setup
#### Prerequisites: 
* [anaconda](https://www.anaconda.com/products/individual)

1. Create a virtual environment
```bash
conda create --name value python=3.6.9
conda activate value
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Install spaCy English pipeline and nltk wordnet
```python
python -m spacy download en_core_web_sm
python 
>>> import nltk
>>> nltk.download('wordnet')
>>> nltk.download('cmudict')
>>> quit()
```

4. Confirm that your setup is correct by running the unittest
```bash
python -m unittest tests.py
```

#### Build Multi-VALUE CoQA (optional)
1. Pull data
```bash
bash pull_coqa.sh
```

2. Run for each dialect
```bash
python -m src.build_coqa_value --dialect aave &
python -m src.build_coqa_value --dialect appalachian &
python -m src.build_coqa_value --dialect chicano &
python -m src.build_coqa_value --dialect indian &
python -m src.build_coqa_value --dialect multi &
python -m src.build_coqa_value --dialect singapore &
```
## Training and Finetuning Code
All the training and finetuning code can be found under the folder:

    boda/
Some of the code is Jupyter Notebooks because it allows us to experiment more quickly (to create vizualizations, run experiments, etc.)

## Models 
Due to the sheer number of ablation experiments we run it is impractical due to storage and time constraints to save all the models and upload them. Hence, many models are just used to run baseline evalaution and are not saved. The adapters we train are saved but it is difficult to share them. Hence, we do not include the models explicitly right now. For the final report, we will include the best performing model to run after we are finished with experimentation.

## Data
All the data used is available and uploaded publically on huggingface. The links are found below:

 - Copa: https://huggingface.co/datasets/super_glue/viewer/copa
 - FigQA: https://huggingface.co/datasets/nightingal3/fig-qa
 - WikiLingua: https://huggingface.co/datasets/GEM/wiki_lingua/viewer/en

For the Dialectal versions we generated:

- Dia_Copa: https://huggingface.co/datasets/ashabrawy/dia_copa
- Dia_FigQA: https://huggingface.co/datasets/ashabrawy/dia_figqa
- Dia_WikiLingua: https://huggingface.co/datasets/ashabrawy/dia_wikilingua
