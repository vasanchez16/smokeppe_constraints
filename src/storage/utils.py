import pandas as pd


def save_dataset(dataset, filename):
    dataset.to_csv(filename, index=True)


def save_indexed_dataset(dataset, index, filename):
    dataset = dataset.iloc[:index]
    save_dataset(dataset, filename)
