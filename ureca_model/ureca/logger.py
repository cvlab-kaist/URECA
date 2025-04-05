# from transformers.integrations import WandbCallback
# import pandas as pd
# import numpy as np
# from internvl.train.dataset import random_split
# from transformers import TrainerCallback
#
# def slice_unknown_vocab(labels_ids, predictions):
#     selected = labels_ids != -100
#     prediction_selected = np.concatenate([selected[:, 1:], np.zeros((selected.shape[0], 1), dtype=bool)], axis=1)
#
#     new_labels_ids, new_predictions = [], []
#     for i, (label_id, prediction) in enumerate(zip(labels_ids, predictions)):
#         new_labels_ids.append(label_id[selected[i]])
#         new_predictions.append(prediction[prediction_selected[i]])
#
#     return new_labels_ids, new_predictions
#
# def decode_predictions(tokenizer, predictions):
#     prediction_ids = predictions.predictions.argmax(axis=-1)
#     labels_ids, prediction_ids = slice_unknown_vocab(predictions.label_ids, prediction_ids)
#     labels = tokenizer.batch_decode(labels_ids)
#     prediction_text = tokenizer.batch_decode(prediction_ids)
#
#     return {"labels": labels, "predictions": prediction_text}
#
#
# class WandbPredictionProgressCallback(WandbCallback):
#     """Custom WandbCallback to log model predictions during training.
#
#     This callback logs model predictions and labels to a wandb.Table at each
#     logging step during training. It allows to visualize the
#     model predictions as the training progresses.
#
#     Attributes:
#         trainer (Trainer): The Hugging Face Trainer instance.
#         tokenizer (AutoTokenizer): The tokenizer associated with the model.
#         sample_dataset (Dataset): A subset of the validation dataset
#           for generating predictions.
#         num_samples (int, optional): Number of samples to select from
#           the validation dataset for generating predictions. Defaults to 100.
#         freq (int, optional): Frequency of logging. Defaults to 2.
#     """
#
#     def __init__(self, trainer, tokenizer, val_dataset, num_samples=100, freq=2):
#         """Initializes the WandbPredictionProgressCallback instance.
#
#         Args:
#             trainer (Trainer): The Hugging Face Trainer instance.
#             tokenizer (AutoTokenizer): The tokenizer associated
#               with the model.
#             val_dataset (Dataset): The validation dataset.
#             num_samples (int, optional): Number of samples to select from
#               the validation dataset for generating predictions.
#               Defaults to 100.
#             freq (int, optional): Frequency of logging. Defaults to 2.
#         """
#         super().__init__()
#         self.trainer = trainer
#         self.tokenizer = tokenizer
#
#         total_size = len(val_dataset)
#         train_size = total_size - num_samples
#
#         _, test_dataset = random_split(val_dataset, [train_size, num_samples])
#
#         self.sample_dataset = test_dataset
#         self.freq = freq
#
#     def on_evaluate(self, args, state, control, **kwargs):
#         super().on_evaluate(args, state, control, **kwargs)
#         # control the frequency of logging by logging the predictions
#         # every `freq` steps
#         if state.global_step % self.freq == 0:
#             # generate predictions
#             predictions = self.trainer.predict(self.sample_dataset)
#             # decode predictions and labels
#             predictions = decode_predictions(self.tokenizer, predictions)
#             # add predictions to a wandb.Table
#             predictions_df = pd.DataFrame(predictions)
#             predictions_df["step"] = state.global_step
#             records_table = self._wandb.Table(dataframe=predictions_df)
#             # log the table to wandb
#             self._wandb.log({f"sample_predictions_{str(state.global_step)}": records_table})
#
# class SaveOnEpochEndCallback(TrainerCallback):
#     def on_epoch_end(self, args, state, control, **kwargs):
#         # Force saving at the end of each epoch
#         control.should_save = True
#         return control


from transformers.integrations import WandbCallback
import pandas as pd
import numpy as np
# import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from internvl.train.dataset import random_split
from transformers import TrainerCallback
from nltk.tokenize import word_tokenize


def slice_unknown_vocab(labels_ids, predictions):
    selected = labels_ids != -100
    prediction_selected = np.concatenate([selected[:, 1:], np.zeros((selected.shape[0], 1), dtype=bool)], axis=1)

    new_labels_ids, new_predictions = [], []
    for i, (label_id, prediction) in enumerate(zip(labels_ids, predictions)):
        new_labels_ids.append(label_id[selected[i]][:-1])
        new_predictions.append(prediction[prediction_selected[i]][:-1])

    return new_labels_ids, new_predictions


def decode_predictions(tokenizer, predictions):
    prediction_ids = predictions.predictions.argmax(axis=-1)
    labels_ids, prediction_ids = slice_unknown_vocab(predictions.label_ids, prediction_ids)
    labels = tokenizer.batch_decode(labels_ids)
    prediction_text = tokenizer.batch_decode(prediction_ids)
    return {"labels": labels, "predictions": prediction_text}


def compute_metrics(labels, predictions):
    bleu_weights =[(1, 0, 0, 0), (0.5, 0.5, 0, 0),
                   (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]

    bleu_scores_1, bleu_scores_2, bleu_scores_3, bleu_scores_4 =[], [], [], []
    meteor_scores, rouge_scores = [], []
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for ref, pred in zip(labels, predictions):
        ref_tokens = word_tokenize(ref)
        pred_tokens = word_tokenize(pred)

        bleu_scores_1.append(sentence_bleu(ref_tokens, pred_tokens, weights=bleu_weights[0]))
        bleu_scores_2.append(sentence_bleu(ref_tokens, pred_tokens, weights=bleu_weights[1]))
        bleu_scores_3.append(sentence_bleu(ref_tokens, pred_tokens, weights=bleu_weights[2]))
        bleu_scores_4.append(sentence_bleu(ref_tokens, pred_tokens, weights=bleu_weights[3]))
        meteor_scores.append(meteor_score([ref_tokens], pred_tokens))
        rouge_scores.append(scorer.score(ref, pred)['rougeL'].fmeasure)

    return {
        "bleu-1": np.mean(bleu_scores_1),
        "bleu-2": np.mean(bleu_scores_2),
        "bleu-3": np.mean(bleu_scores_3),
        "bleu-4": np.mean(bleu_scores_4),
        "meteor": np.mean(meteor_scores),
        "rougeL": np.mean(rouge_scores),
    }


class WandbPredictionProgressCallback(WandbCallback):
    def __init__(self, trainer, tokenizer, val_dataset, num_samples=100, freq=2):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        total_size = len(val_dataset)
        train_size = total_size - num_samples
        _, test_dataset = random_split(val_dataset, [train_size, num_samples])
        self.sample_dataset = test_dataset
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        if state.global_step % self.freq == 0:
            predictions = self.trainer.predict(self.sample_dataset)
            decoded = decode_predictions(self.tokenizer, predictions)
            metrics = compute_metrics(decoded["labels"], decoded["predictions"])

            predictions_df = pd.DataFrame(decoded)
            predictions_df["step"] = state.global_step
            records_table = self._wandb.Table(dataframe=predictions_df)

            self._wandb.log({
                f"sample_predictions_{str(state.global_step)}": records_table})

            self._wandb.log({
                "eval/bleu_score@1": metrics["bleu-1"],
                "eval/bleu_score@2": metrics["bleu-2"],
                "eval/bleu_score@3": metrics["bleu-3"],
                "eval/bleu_score@4": metrics["bleu-4"],
                "eval/meteor_score": metrics["meteor"],
                "eval/rougeL_score": metrics["rougeL"],
            }, step=state.global_step)


class SaveOnEpochEndCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = True
        return control
