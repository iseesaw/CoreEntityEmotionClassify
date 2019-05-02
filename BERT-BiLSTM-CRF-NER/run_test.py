import re
import os
from pyltp import Segmentor
import json

def loadStopwords():
    stopwords = set([])
    with open("stopwords.txt", "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            stopwords.add(line)
    return stopwords

def read():
    # LTP_DATA_DIR = "3.4.0/ltp_data_v3.4.0"
    # cws_model_path = os.path.join(LTP_DATA_DIR, "cws.model")
    # segmentor = Segmentor()
    # segmentor.load(cws_model_path)

    # stopwords = loadStopwords()
    m = 1000
    with open("data/coreEntityEmotion_train.txt", "r", encoding="utf-8") as f:
        length = 0
        for line in f.readlines()[:1]:
            line = line.strip()
            data = json.loads(line)
            
            content = data["content"]
            if len(content) < m:
                m = len(content)
    print(m)

def saveJson(data):
    with open("train.txt", "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False))


if __name__ == "__main__":
    read()