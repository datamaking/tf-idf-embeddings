import psutil
import time
import pandas as pd

def monitor_resources(duration=5):
    process = psutil.Process()

    i = 0
    for _ in range(duration):

        if i == 2:
            df = pd.read_csv("/home/datamaking/work/code/workarea/django-project/data/ag_news.csv")
            print(df.shape)

        if i == 4:
            print(df.shape)
            del df

        mem = process.memory_info().rss / (1024 * 1024)  # in MB
        cpu = process.cpu_percent(interval=1)
        print(f"Memory usage: {mem:.2f} MB, CPU usage: {cpu:.2f}%")
        time.sleep(1)
        i = i + 1

if __name__ == "__main__":
    monitor_resources()
