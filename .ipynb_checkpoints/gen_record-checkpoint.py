from src import Dialects as value
from datasets import load_dataset, Dataset, DatasetDict
import os
import yaml

dialects = [dialect for dialect in dir(value) if dialect[0] == dialect[0].upper() and dialect[0].isalpha() and dialect != "BaseDialect" and dialect != "MultiDialect"][1:]
dia_codes = ['ABO', 'AFR', 'APP', 'AUS', 'AU1', 'BAH', 'BLA', 'CAM', 'CAP', 'CHA', 'CHI', 'COL', 'CO1', 'DIA', 'DI1', 'EAR', 'EAS', 'FAL', 'FIJ', 'FI1', 'GHA', 'HON', 'IND', 'IN1', 'IRI', 'JAM', 'KEN', 'LIB', 'MAL', 'MA1', 'MAN', 'NEW', 'NE1', 'NIG', 'NOR', 'ORK', 'OZA', 'PAK', 'PHI', 'RUR', 'SCO', 'SOU', 'SO1', 'SO2', 'SRI', 'STH', 'TAN', 'TRI', 'UGA', 'WEL', 'WHI', 'WH1']

ds = load_dataset("super_glue", "record", cache_dir="/scratch/afz225/.cache")

ds['train'] = Dataset.from_dict(ds['train'][:1000])
ds['validation'] = Dataset.from_dict(ds['validation'][:500])
ds['test'] = Dataset.from_dict(ds['test'][:500])

def map_record(example):
    example["passage"] = dia.transform(example["passage"])
    example["query"] = dia.transform(example["query"])
    return example

try:
    os.mkdir(f"dia_record")
except:
    print("dir_exists")
for i, dialect in enumerate(dialects):
    dia = getattr(value, dialect)()
    # try:
    dia_ds = ds.map(map_record, num_proc=100)
    # except:
    #     continue
    for split in dia_ds:
        try:
            os.mkdir(f"dia_record/{dia_codes[i]}")
        except:
            print("dir_exists")
        dia_ds[split].to_csv(f"dia_record/{dia_codes[i]}/{split}.csv")

with open("dia_record/README.md", "w") as f:
    f.write("---\n")
    configs = []
    for dialect in dia_codes:
        data_files = [
                {
                  "path": f"{dialect}/{split}.csv", 
                  "split": split
                } 

              for split in ds]
        configs.append({"data_files": data_files, 
              "config_name": dialect.lower()})
    
    yaml.dump({
        "configs": configs
    }, f)
    f.write("---\n")
