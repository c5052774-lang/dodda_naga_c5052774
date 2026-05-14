import tensorflow as tf
from transformers import TFAutoModel


class BertSequenceClassifier(tf.keras.Model):

    def __init__(self, model_name, num_labels=5,
                 dropout_rate=0.1,
                 freeze_bert=False):

        super().__init__()

        # Try to load TF weights first; if that fails, attempt loading PyTorch
        # weights and converting them to TF with `from_pt=True`.
        try:
            self.bert = TFAutoModel.from_pretrained(model_name)
        except Exception as e:
            try:
                self.bert = TFAutoModel.from_pretrained(model_name, from_pt=True)
            except Exception as e2:
                raise RuntimeError(
                    "Failed to load transformer model '{0}'. First attempt: {1}. "
                    "Second attempt with from_pt=True: {2}. "
                    "This often indicates an incompatible `transformers`/`safetensors`/" \
                    "`tensorflow` combination. Try pinning `transformers` to a 4.x "
                    "release (e.g. `pip install transformers==4.33.2 safetensors`) or "
                    "switching to PyTorch model classes.".format(model_name, e, e2)
                ) from e2

        if freeze_bert:
            self.bert.trainable = False

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.classifier = tf.keras.layers.Dense(
            num_labels,
            activation="softmax"
        )

    def call(self, inputs, training=False):

        input_ids, attention_mask = inputs

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=training
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output, training=training)

        return self.classifier(pooled_output)