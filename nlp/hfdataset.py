import requests
import time
import math
from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm
from datasets import load_dataset
#from datasets import list_datasets
from huggingface_hub import list_datasets
import pandas as pd

def fetch_issues(
    owner="huggingface",
    repo="datasets",
    num_issues=10_000,
    rate_limit=5_000,
    issues_path=Path("."),
):
    if not issues_path.is_dir():
        issues_path.mkdir(exist_ok=True)

    batch = []
    all_issues = []
    per_page = 100  # Number of issues to return per page
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    for page in tqdm(range(num_pages)):
        # Query with state=all to get both open and closed issues
        query = f"issues?page={page}&per_page={per_page}&state=all"
        issues = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
        batch.extend(issues.json())

        if len(batch) > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            batch = []  # Flush batch for next time period
            print(f"Reached GitHub rate limit. Sleeping for one hour ...")
            time.sleep(60 * 60 + 1)

    all_issues.extend(batch)
    df = pd.DataFrame.from_records(all_issues)
    df.to_json(f"{issues_path}/{repo}-issues.jsonl", orient="records", lines=True)
    print(
        f"Downloaded all the issues for {repo}! Dataset stored at {issues_path}/{repo}-issues.jsonl"
    )

if __name__ == "__main__":

    all_datasets = list_datasets()
    all_datasetids = [dataset.id for dataset in all_datasets]
    print(all_datasetids)
    print(f"There are {len(all_datasetids)} datasets currently available on the Hub")
    print(f"The first 10 are: {all_datasets[:10]}")

    # Load a dataset and print the first example in the training set
    squad_dataset = load_dataset('squad')
    print(squad_dataset['train'][0])

    # dataset_id="amazon_reviews_multi"
    # dataset_config="all_languages"
    # dataset = load_dataset(dataset_id,dataset_config)

    emotions = load_dataset("emotion")
    print(emotions)
    train_ds = emotions["train"]
    print(len(train_ds))
    print(train_ds[0])#single example
    print(train_ds.column_names)#print column names
    print(train_ds.features)
    #Datasets is based on Apache Arrow, which defines a typed columnar format that is more memory efficient than native Python.

    emotions.set_format(type="pandas")
    df = emotions["train"][:]
    print(df.head())

    url = "https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=1"
    response = requests.get(url)
    print(response.status_code)
    print(response.json())

    fetch_issues()

    issues_dataset = load_dataset("json", data_files="datasets-issues.jsonl", split="train")
    issues_dataset