from src import Dialects as value
from datasets import load_dataset, Dataset, DatasetDict
import os
import yaml

# dialects = [dialect for dialect in dir(value) if dialect[0] == dialect[0].upper() and dialect[0].isalpha() and dialect != "BaseDialect" and dialect != "MultiDialect"][25:]
dialects = ["AustralianDialect", "HongKongDialect", "ColloquialAmericanDialect"]
dia_codes = ["AUS", "HON", "COL"]

ds = load_dataset("GEM/wiki_lingua", "en", cache_dir="/scratch/afz225/.cache")

ds['train'] = Dataset.from_dict(ds['train'][:500])
ds['validation'] = Dataset.from_dict(ds['validation'][:100])
ds['test'] = Dataset.from_dict(ds['test'][:100])

def map_wikilingua(example):
    example["passage"] = dia.transform(example["source"])
    example["query"] = dia.transform(example["target"])
    return example
try:
    os.mkdir(f"dia_wikilingua")
except:
    print("dir_exists")

for i, dialect in enumerate(dialects):
    dia = getattr(value, dialect)()
    # try:
    dia_ds = ds.map(map_wikilingua, num_proc=100)
    # except:
    #     continue
    for split in ["train", "test", "validation"]:
        try:
            os.mkdir(f"dia_wikilingua/{dia_codes[i]}")
        except:
            print("dir_exists")
        dia_ds[split].to_csv(f"dia_wikilingua/{dia_codes[i]}/{split}.csv")
        

with open("dia_wikilingua/README.md", "w") as f:
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
