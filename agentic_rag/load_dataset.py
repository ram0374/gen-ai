import datasets


def get_raw_data():
    return datasets.load_dataset("m-ric/huggingface_doc", split="train")

knowledge_base = get_raw_data()