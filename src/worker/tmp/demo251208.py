# Used for 2025/12/08 demo only, worker side
# Load model from /user/pi/fyp/model and run inference server
# Evaluate performance and log results
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# The demonstration model (demo/251208demo) is a finetuned pretrained BERT model for my BC3415 course, following the IMDB sentiment analysis tutorial on HuggingFace
# Colab Notebook: https://colab.research.google.com/drive/1jrFNsHAsZ4cq5bDLJQ_RPAFjgMOC0hfB?usp=sharing
# Github Repo: https://github.com/SyntaxaR/BC3415_IA/
# Dataset: https://www.kaggle.com/datasets/mehmetisik/amazon-review/data
# Intake: reviews for an Sandisk SD card on Amazon
# Output: sentiment classification (negative/0, neutral/1, positive/2)
# Accuracy: ~95% on test set

dataset2id = {1:0, 2:0, 3:1, 4:2, 5:2}
dataset2label = {1:"NEGATIVE", 2:"NEGATIVE", 3:"NEUTRAL", 4:"POSITIVE", 5:"POSITIVE"}
id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2:"POSITIVE"}
label2id = {"NEGATIVE": 0, "NEUTRAL": 1,"POSITIVE": 2}

model_dir = "/home/pi/fyp/model/"


class InferenceServer:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        self.count = 0

    # Input: [Combined text (title + review, truncation needed), dataset output id (0/1/2)]
    def run_inference(self, text: str, true_id: int):
        pred_label = self.classifier(text[:self.tokenizer.model_max_length])[0]['label']
        pred_id = label2id[pred_label]
        self.count += 1
        if self.count % 50 == 0:
            print(f"Every 50: Input Text: {text[:30]}... | Predicted: {pred_label} | True: {id2label[dataset2id[true_id]]}")
            self.count = 0
        return pred_id == true_id