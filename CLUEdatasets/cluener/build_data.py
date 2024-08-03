import json

label_dict = dict()
with open('new_train.json', 'w', encoding='utf-8') as fp:
    for line in open('train.json', 'r', encoding='utf-8').readlines():
        data = json.loads(line.strip())

        # {"text": "邓颖超自然出席，宋美龄应邀到会并讲话。", "entity_list": [{"type": "nr", "argument": "邓颖超"}, {"type": "nr", "argument": "宋美龄"}]}
        entity_list = []
        for k, v in data['label'].items():
            if k not in label_dict:
                label_dict[k] = len(label_dict)
            for k1, v1 in v.items():
                entity_list.append({"type": k, "argument": k1})
        print(json.dumps({"text": data["text"],
                          "entity_list": entity_list}, ensure_ascii=False),
              file=fp)

with open('new_dev.json', 'w', encoding='utf-8') as fp:
    for line in open('dev.json', 'r', encoding='utf-8').readlines():
        data = json.loads(line.strip())

        # {"text": "邓颖超自然出席，宋美龄应邀到会并讲话。", "entity_list": [{"type": "nr", "argument": "邓颖超"}, {"type": "nr", "argument": "宋美龄"}]}
        entity_list = []
        for k, v in data['label'].items():
            assert k in label_dict
            for k1, v1 in v.items():
                entity_list.append({"type": k, "argument": k1})
        print(json.dumps({"text": data["text"],
                          "entity_list": entity_list}, ensure_ascii=False),
              file=fp)

json.dump(label_dict, open('label_dict.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
