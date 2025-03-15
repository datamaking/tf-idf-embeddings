import cProfile
import pstats
import io
import pandas as pd
import time


def process_text_corpus(corpus):
    df = pd.read_csv("/home/datamaking/work/code/workarea/django-project/data/ag_news.csv")
    print("Before sleep ... ")
    time.sleep(10)
    print(df.memory_usage().sum())
    print("After the sleep.")
    print(df.shape)
    print(df.memory_usage(deep=True).sum())

    # Simulate an NLP task (e.g., tokenization, vectorization, etc.)
    tokens = [doc.lower().split() for doc in corpus]
    # Simulate further processing (dummy delay)
    for doc in tokens:
        [word[::-1] for word in doc]
    return tokens


if __name__ == "__main__":
    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "Never jump over the lazy dog quickly",
        "Bright foxes leap over lazy dogs in summer"
    ]

    # Run cProfile on the process_text_corpus function
    profiler = cProfile.Profile()
    profiler.enable()
    process_text_corpus(corpus)
    profiler.disable()

    # Print profiling results sorted by cumulative time
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats(10)
    print(s.getvalue())
