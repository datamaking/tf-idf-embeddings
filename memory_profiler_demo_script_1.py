# pip install memory_profiler

from memory_profiler import profile
import pandas as pd

@profile
def process_text_corpus(corpus):
    df = pd.read_csv("/home/datamaking/work/code/workarea/django-project/data/ag_news.csv")
    print(df.shape)
    print(df.info())
    print(df.info(memory_usage='deep'))
    tokens = [doc.lower().split() for doc in corpus]
    # Simulate additional processing
    processed = []
    for doc in tokens:
        processed.append([word[::-1] for word in doc])

    del df
    return processed

if __name__ == "__main__":
    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "Never jump over the lazy dog quickly",
        "Bright foxes leap over lazy dogs in summer"
    ]
    process_text_corpus(corpus)
