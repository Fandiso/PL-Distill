# -*- coding: utf-8 -*-
import argparse
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import json
import os
import torch
import librosa
from io import BytesIO
from urllib.request import urlopen
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score, f1_score


def ComputePerformance(ref_id, hyp_id):

    score = dict()
    num = len(ref_id)

    score['num'] = num
    score['overallWA'] = accuracy_score(ref_id, hyp_id)
    score['overallUA'] = balanced_accuracy_score(ref_id, hyp_id)
    score['overallMicroF1'] = f1_score(ref_id, hyp_id, average='micro')
    score['overallMacroF1'] = f1_score(ref_id, hyp_id, average='macro')
    score['report'] = classification_report(ref_id, hyp_id)
    score['confusion'] = confusion_matrix(ref_id, hyp_id)
    return score


###### Write scores into file
def WriteScore(f, score):
    classification = 'Emotion Recognition'
    K = score['num']
    f.write('%sScoring -- Sample [%d], Overall UA     %.4f\n' % (classification, K, score['overallUA']))
    f.write('%sScoring -- Sample [%d], Overall WA     %.4f\n' % (classification, K, score['overallWA']))
    f.write('%sScoring -- Sample [%d], Overall Micro-F1     %.4f\n' % (classification, K, score['overallMicroF1']))
    f.write('%sScoring -- Sample [%d], Overall Macro-F1     %.4f\n' % (classification, K, score['overallMacroF1']))
    f.write('\n')
    f.write(score['report'])
    f.write('\n')
    confusion_mtx = score['confusion']
    f.write(f'confusion_matrix: \n {confusion_mtx}')


class EmoEval():
    def __init__(self, predictions, data_labels):
        """
            predictions: [{"key": "xxx", pred":1}, {"key":"yyy",pred":2},...]
            data_labels: [{"key":"xxx",label":1}, {"key":"yyy",label":3}, ...]
        """
        assert len(predictions) == len(data_labels), f'the number of predictions shoud be equal to labels'
        for pred, label in zip(predictions, data_labels):
            pred_key = pred['key']
            label_key = label['key']
            assert pred_key == label_key, f'prediction and label should have the same key, while prediction has a key {pred_key}, label has a key {label_key}'

        self.targets = []
        for instance in data_labels:
            label = instance['label']
            self.targets.append(label)

        self.predictions = []
        for instance in predictions:
            pred = instance['pred']
            self.predictions.append(pred)

    def compute_metrics(self, ):
        scores = ComputePerformance(self.targets, self.predictions)
        return scores

    def write_scores(self, path, scores):
        with open(path, "w") as f:
            WriteScore(f, scores)


class Qwen2AudioEvaluateDataset(Dataset):
    """
    输入：data_path, processor
    输出：inputs,labels,audio_key
    """
    def __init__(self, data_path: str, processor, data_type):

        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))  # 按行读取并解析

        self.processor = processor
        self.sr = processor.feature_extractor.sampling_rate  # 官方示例同款
        self.data_type = data_type

    def _load_one(self, path: str):
        if path.startswith("http://") or path.startswith("https://"):
            wav, _ = librosa.load(BytesIO(urlopen(path).read()), sr=self.sr)
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            wav, _ = librosa.load(path, sr=self.sr)
        return wav

    def _load_audios(self, audio_field):
        if not audio_field:  # None / "" / []
            return []
        if isinstance(audio_field, str):
            return [self._load_one(audio_field)]
        # list[str]
        return [self._load_one(p) for p in audio_field]

    def __len__(self): return len(self.data)

    def __getitem__(self, i):

        ex = self.data[i]

        if self.data_type == "iemocap":
            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": "<|AUDIO|>Please identify the emotion of this audio clip, then choose one word from [Happy, Sad, Angry, Neutral] as your answer."},
                ]}
            ]
        elif self.data_type == "ravdess":
            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": "<|AUDIO|>Please identify the emotion of this audio clip, then choose one word from [Happy, Sad, Angry, Neutral, Disgust, Surprise, Fear, Calm] as your answer."},
                ]}
            ]
        elif self.data_type == "savee":
            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": "<|AUDIO|>Please identify the emotion of this audio clip, then choose one word from [Happy, Sad, Angry, Neutral, Disgust, Surprise, Fear] as your answer."},
                ]}
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audio_path = ex.get("audios", [])
        label = ex.get("response", "")
        audio_key = os.path.basename(audio_path[0] if type(audio_path) == list else audio_path)

        # 加载音频
        audio = self._load_audios(audio_path)

        # 构造完整输入
        inputs = self.processor(
            text=prompt,
            audios=audio,
            return_tensors="pt",
            sampling_rate=self.processor.feature_extractor.sampling_rate
        )

        return inputs,label,audio_key


def model_pred(model, inputs, true_label=None, audio_key=None, prediction_path=None):
    """
    预测音频情绪

    输入:
    model:模型
    inputs: 输入数据
    true_label: 可选的真实标签
    audio_key: 音频标识符
    prediction_path: 预测结果保存路径

    返回:
    response :输出结果
    """
    # 确保设备一致
    device = model.device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    # 生成预测
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=256)
        generate_ids = generate_ids[:, inputs['input_ids'].size(1):]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        response = response.strip()

    #结果写到文件里
    with open(prediction_path, "a") as f:
        if true_label is not None:
            f.write(f"{audio_key}\t{response}\t{true_label}\n")
        else:
            f.write(f"{audio_key}\t{response}\n")

    return response



def evaluate(dataset, model, prediction_path, scores_path):
    """
    在给定的数据集上评估模型的性能

    参数:
    dataset: 用于评估的数据集
    model: 预训练模型
    processor: 数据预处理器
    label2idx: 标签映射字典

    返回:
    评估结果字典
    """
    fold_test_pred = []
    fold_test_targets = []
    # 在主循环中修改调用方式，传入真实标签
    for inputs,label,audio_key in dataset:
        #生成答案
        response = model_pred(model ,inputs, true_label=label, audio_key=audio_key , prediction_path=prediction_path)
        pred_index = map_emotion(response)
        label_index = map_emotion(label)
        fold_test_pred.append({"key": audio_key, "pred": pred_index})
        fold_test_targets.append({"key": audio_key, "label": label_index})
    # 使用 EmoEval 计算评估指标
    fold_emo_eval = EmoEval(fold_test_pred, fold_test_targets)
    fold_scores = fold_emo_eval.compute_metrics()
    fold_emo_eval.write_scores(scores_path, fold_scores)

    return fold_scores

#情感向数字标签映射
def map_emotion(e):

    prefix_rules = {
        "Neutral": 0,
        "Happy": 1,
        "Sad": 2,
        "Ang": 3,
        "Sur": 4,
        "Dis": 5,
        "Fear": 6,
    }

    emotion_to_idx = {
        'neutral': 0, 'Neutral': 0, 'NEUTRAL': 0,
        'happy': 1, 'Happy': 1, 'HAPPY': 1,'joy': 1, 'Joy': 1, 'JOY': 1,
        'sad': 2, 'Sad': 2, 'SAD': 2,'sadness': 2, 'Sadness': 2, 'SADNESS': 2,
        'angry': 3, 'Angry': 3, 'ANGRY': 3,'anger':3, 'Anger': 3, 'ANGER': 3,
        'surprise': 4, 'Surprise': 4, 'SURPRISE': 4,'surprised': 4, 'Surprised': 4, 'SURPRISED': 4,'Surprprise':4, 'Surprpr':4,'Surprprpr':4,'Surprprprise':4,
        'disgust': 5, 'Disgust': 5, 'DISGUST': 5,'Disgusted': 5, 'disgusted': 5,
        'fear': 6, 'Fear': 6, 'FEAR': 6, 'fearful': 6,'Fearful': 6,'FEARFUL': 6,
    }
    e = e.strip()
    # 先尝试精确匹配
    if e in emotion_to_idx:
        return emotion_to_idx[e]
    # 再尝试前缀匹配
    for prefix, idx in prefix_rules.items():
        if e.startswith(prefix):
            return idx
    # 如果都不匹配，默认 -1 或者丢弃
    return 0

def build_argparser():

    parser = argparse.ArgumentParser(description="Evaluate Qwen2-Audio-0.5B model")
    # 如果你偶尔也想通过命令行切换数据/模型路径，可以把下面几项也暴露出来
    parser.add_argument("--prediction_path", type=str, required=True, help="预测结果与真实结果保存路径")
    parser.add_argument("--scores_path", type=str, required=True, help="分数保存路径")
    parser.add_argument("--lora_path", type=str, required=True, help="待测检查点加载路径")
    parser.add_argument("--data_path", type=str, required=True, help="测试数据路径")
    parser.add_argument("--data_type", type=str, default="ravdess", choices=["iemocap", "ravdess", "savee"], help="数据集类型")
    return parser


if __name__ == "__main__":

    args_cli = build_argparser().parse_args()

    # 1.加载预训练模型和处理器
    original_model_path = "./Qwen2-Audio-0.5B"
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        original_model_path,
        device_map={"":0},
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(original_model_path)

    # 2.加载训练好的lora检查点和数据
    lora_path = args_cli.lora_path
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    data_path = args_cli.data_path
    Eval_dataset = Qwen2AudioEvaluateDataset(data_path, processor, args_cli.data_type)

    # 3.定义结果保存路径
    pre_dir = os.path.dirname(args_cli.prediction_path)
    if not os.path.exists(pre_dir):
        os.makedirs(pre_dir)
    sco_dir = os.path.dirname(args_cli.scores_path)
    if not os.path.exists(sco_dir):
        os.makedirs(sco_dir)
    prediction_path = args_cli.prediction_path
    scores_path = args_cli.scores_path

    # 4.评估模型
    scores = evaluate(dataset = Eval_dataset, model=model, prediction_path=prediction_path, scores_path=scores_path)
    print(f"预测结果保存在{prediction_path}，评估分数保存在{scores_path}")
