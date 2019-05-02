import json
import re


def create_example(entry):
    data_list = []

    #entry = json.loads(line,strict=False)
    entities = entry['coreEntityEmotions']
    entity_set = set()
    for entity in entities:
        entity_name = entity['entity']
        entity_set.add(entity_name)
    content = entry['title']+entry['content']
    lines = re.split(r"\n", content)
    for line in lines:
        if not line:
            continue
        sentences = re.split(r"([。])", line)
        sentences.append("")
        sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
        order = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence == "" or sentence == '\r' or len(sentence) == 1 or len(sentence) > 500:
                continue
            order += 1
            label_result = []
            seg_result = []
            beginset = set()
            endset = set()
            for name in entity_set:
                listb, liste = search(name, sentence)
                for b in listb:
                    beginset.add(b)
                for e in liste:
                    endset.add(e)
            state = 0
            for index in range(len(sentence)):
                seg_result.append(sentence[index])
                if index in beginset:
                    label_result.append('B')
                    state = 1
                    continue
                if index in endset:
                    label_result.append('I')
                    state = 0
                    continue
                if index not in beginset and index not in endset and state == 1:
                    label_result.append('I')
                    continue

                label_result.append('O')
            subjson = dict()
            subjson['newsId'] = entry['newsId']+"_"+str(order)
            subjson['seq'] = seg_result
            subjson['label'] = label_result
            data_list.append(subjson)
    return data_list


def search(subtext, string: str):
    list = []
    liste = []
    text = string
    index = 0
    while True:
        index = text.find(subtext, index)
        if index == -1:
            return list,liste
        else:
            list.append(index)
            liste.append(index+len(subtext)-1)
            index += len(subtext)


if __name__ == "__main__":
    # 例子
    string = '{"newsId": "7bdc768b", "coreEntityEmotions": [{"entity": "abc", "emotion": "NORM"}, ' \
             '{"entity": "d", "emotion": "NORM"}, {"entity": "e", "emotion": "NORM"}], ' \
             '"title": "abcde", "content": "abcdefadf\nadsea\n"}'
    print(create_example(string))