import json
import os
folds = 5
data_list = []
output_list = []

label_map = {'hap':'Happy', 'sad':'Sad', 'ang':'Angry', 'neu':'Neutral'}  # you may need to define a label to index mapping for your own training, see `data/iemocap/label_map.json`

for i in range (folds):
    train_data_path = f"/PL-Distill/EmoBox/data/iemocap/fold_{i+1}/iemocap_train_fold_{i+1}.jsonl" # original train file path
    train_output_path = f"/PL-Distill/EmoBox/data/iemocap_train_fold_{i+1}.jsonl"  #  your train file path
    test_data_path = f"/PL-Distill/EmoBox/data/iemocap/fold_{i+1}/iemocap_test_fold_{i+1}.jsonl" #original test file path
    test_output_path = f"/PL-Distill/EmoBox/data/iemocap_test_fold_{i+1}.jsonl"  #  your test file pat
    # valid_data_path = f""
    # valid_output_path = f""
    data_list.append(train_data_path)
    output_list.append(train_output_path)
    data_list.append(test_data_path)
    output_list.append(test_output_path)
    # data_list.append(valid_data_path)
    # output_list.append(valid_output_path)

for i in range(len(data_list)):
    data_path = data_list[i]
    output_path = output_list[i]
    data = []

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    if  not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            entry['audios'] = entry['wav']
            entry['audios'] = entry['audios'].replace('downloads', '')
            entry['response'] = entry['emo']
            entry['response'] = label_map[entry['response']]
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')  # 每个 JSON 对象写入新的一行
    print(f"Data saved to new file: {output_path}")



