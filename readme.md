# DistilBERT Classification with IMDb Review Data

This repository contains code for fine-tuning the DistilBERT model for text classification using IMDb review data. The DistilBERT model, a lightweight version of BERT, has been trained on a large corpus of text and can be fine-tuned for various NLP tasks, including classification.

## Model Description

DistilBERT is a transformer-based model that has been pre-trained on a large amount of text data. This repository focuses on fine-tuning DistilBERT for binary text classification using IMDb review data. The IMDb dataset consists of movie reviews labeled as positive or negative sentiment. By fine-tuning the DistilBERT model on this dataset, we aim to create a classifier that can predict the sentiment of a given movie review.

## File Structure

The repository has the following file structure:

```
- distillbert_text_classification.ipynb
- pretrained/
  - config.json
  - special_tokens_map.json
  - tf_model.h5
  - tokenizer_config.json
  - tokenizer.json
  - vocab.txt
- README.md
```

- `distillbert_text_classification.ipynb` is a Jupyter notebook that demonstrates how to load the fine-tuned DistilBERT model and perform text classification.
- The `pretrained` directory contains the DistilBERT model files after fine-tuning:
  - `config.json`: Model configuration.
  - `special_tokens_map.json`: Mapping of special tokens.
  - `tf_model.h5`: Model weights.
  - `tokenizer_config.json`: Tokenizer configuration.
  - `tokenizer.json`: Tokenizer file.
  - `vocab.txt`: Vocabulary file.
- `README.md` (this file) provides an overview of the repository and instructions.

## Usage

Download the fine-tuned DistilBERT model from the [release page](https://github.com/HarixhKumawat/DistilBERT-classification-imdb_data/releases).

Use the following script to load the model and perform text classification:

```python
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("./pretrained")
model = TFAutoModelForSequenceClassification.from_pretrained("./pretrained")

text = "This was a worst movie. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the not to watch movies."
inputs = tokenizer(text, return_tensors="tf")

logits = model(**inputs).logits

predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
label = model.config.id2label[predicted_class_id]
print("Predicted Label:", label)
```

Replace `"./pretrained"` with the appropriate path to the directory containing the downloaded model files.



*Disclaimer: This repository is created for educational purposes and may contain code derived from publicly available sources.*