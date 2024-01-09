import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--pred", type=str, required=True)
args = parser.parse_args()

label_file = args.label
ans_file = args.pred

answers = [json.loads(q) for q in open(ans_file, "r")]
labels = [json.loads(q)["label"] for q in open(label_file, "r")]
pred_list = []
label_list = []

for answer, label in zip(answers, labels):
    text = answer["answer"]

    # Only keep the first sentence
    if text.find(".") != -1:
        text = text.split(".")[0]

    text = text.lower()
    if "not" in text or "no" in text:
        pred_list.append(0)
    elif "yes" in text or "there is" in text:
        pred_list.append(1)
    else:
        continue
    label_list.append(int(label == "yes"))

# for i in range(len(label_list)):
#     if label_list[i] == "no":
#         label_list[i] = 0
#     else:
#         label_list[i] = 1


# for answer in answers:
#     if answer["answer"] == "no":
#         pred_list.append(0)
#     else:
#         pred_list.append(1)

pos = 1
neg = 0
yes_ratio = pred_list.count(1) / len(pred_list)

TP, TN, FP, FN = 0, 0, 0, 0
for pred, label in zip(pred_list, label_list):
    if pred == pos and label == pos:
        TP += 1
    elif pred == pos and label == neg:
        FP += 1
    elif pred == neg and label == neg:
        TN += 1
    elif pred == neg and label == pos:
        FN += 1

print("TP\tFP\tTN\tFN\t")
print("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))

precision = float(TP) / float(TP + FP)
recall = float(TP) / float(TP + FN)
f1 = 2 * precision * recall / (precision + recall)
acc = (TP + TN) / (TP + TN + FP + FN)
print("Accuracy: {}".format(acc))
print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("F1 score: {}".format(f1))
print("Yes ratio: {}".format(yes_ratio))
