# MML hallucination

## modification made to third-party model

### minigpt

minigpt4/models/minigpt_base.py: MiniGPTBase.preparing_embedding

## result

step=0
TP FP TN FN
1152 458 1042 348
Accuracy: 0.7313333333333333
Precision: 0.715527950310559
Recall: 0.768
F1 score: 0.7408360128617363
Yes ratio: 0.5366666666666666

fixed weight: pos(1), neg(0.1)
TP FP TN FN
854 583 564 318
Accuracy: 0.6114704614057783
Precision: 0.5942936673625608
Recall: 0.7286689419795221
F1 score: 0.6546569566883863
Yes ratio: 0.6196636481241915
