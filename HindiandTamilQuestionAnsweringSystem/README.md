# Hindi-and-Tamil-Question-Answering-System

This contains the code for the result obtained using XLM-RoBERTa on the chaii dataset.

## Dataset:

The chaii-dataset is used for fine-tuning, along with mlqa (MultiLingual Question Answering) and XQuAD (Cross-lingual Question Answering Dataset) datasets on the XLM-RoBERTa model that has been pre-trained on SQuAD2.
The Chaii dataset consists of the following:

<kbd>**id:**</kbd> unique id for each example<br>
<kbd>**context:**</kbd> a paragraph based on which the questions have to be answered<br>
<kbd>**question:**</kbd> the question that has to be answered<br>
<kbd>**answer_start:**</kbd> the index from which the answer starts (only in the train set)<br>
<kbd>**answer_text:**</kbd> the answer in string format (only in the train set)<br>


## Models:

The model used are mBERT, XLM-RoBERTa.

## Result:

m-BERT (pre-trained on SQuADv1.1, finetuned with chaii) gives 0.55 jaccord score
mDeBERTa gives 0.579 jaccord score
mDeBERTa (finetuned with mlqa, xquad, chaii) gives 0.59 jaccord score
XLM-RoBERTa (pre-trained on squadv2, finetuned with chaii) gives 0.586 jaccord score
XLM-RoBERTa (pre-trained on squadv2, finetuned with mlqa, xquad, chaii) gives 0.616 jaccord score
