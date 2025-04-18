from datetime import datetime
from glob import glob
from textwrap import indent

import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

from transformers import BertForSequenceClassification, BertTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput


class Classifier:
    """
    A helper class for iteratively training a BERT model for text classification.
    """

    _num_labels: int
    _model: BertForSequenceClassification
    _tokenizer: BertTokenizerFast

    _optimizer: SGD
    _criterion: CrossEntropyLoss

    def __init__(self, num_labels: int = 2, model_identifier: str = "google-bert/bert-base-uncased", tokenizer_identifier: str = "google-bert/bert-base-uncased"):
        self._num_labels = num_labels

        self._model = BertForSequenceClassification.from_pretrained(model_identifier, num_labels=num_labels)
        self._tokenizer = BertTokenizerFast.from_pretrained(tokenizer_identifier)

        self._optimizer = SGD(self._model.parameters(), lr=2e-5)
        self._criterion = CrossEntropyLoss()
    
    @property
    def num_labels(self):
        """
        The number of classes.
        """
        return self._num_labels

    @property
    def model(self):
        """
        The BERT model.
        """
        return self._model

    @property
    def tokenizer(self):
        """
        The BERT tokenizer.
        """
        return self._tokenizer
    
    @property
    def optimizer(self):
        """
        The optimizer (SGD).
        """
        return self._optimizer
    
    @property
    def criterion(self):
        """
        The loss function (CrossEntropyLoss).
        """
        return self._criterion

    def backward(self, model_logits: torch.Tensor, expected_idx: int):
        """
        Backpropagate the loss. 
        
        The expected class is given by `expected_idx`, it's an integer between 0 (inclusive) and `num_labels` (exclusive).
        """
        assert 0 <= expected_idx < self.num_labels, f"expected_idx must be between 0 (inclusive) and {self.num_labels} (exclusive)"

        expected_tensor = torch.zeros_like(model_logits)
        expected_tensor[0][expected_idx] = 1.0

        self.optimizer.zero_grad()
        self.criterion(model_logits, expected_tensor).backward()
        self.optimizer.step()

    def predict(self, text: str) -> SequenceClassifierOutput:
        """
        Predict the class probabilities for the given text.
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs
    
    @staticmethod
    def save_path():
        """
        The path to save the model to given the current datetime.
        """
        return f"classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def save(self):
        """
        Save the model to the save path.
        """
        self.model.save_pretrained(self.save_path())
    
    @classmethod
    def load_latest(cls, num_labels: int = None, tokenizer_identifier: str = None):
        """
        Load the model from the most recent save.
        """
        latest = sorted(glob("classifier_*"), reverse=True)[0]
        print(f"Loading model from {latest}")

        kwargs = {
            "model_identifier": latest
        }
        if num_labels is not None:
            kwargs["num_labels"] = num_labels
        if tokenizer_identifier is not None:
            kwargs["tokenizer_identifier"] = tokenizer_identifier
        
        return cls(**kwargs)

    def training_loop(self, texts: list, question: str, answers: list[str], save_after: int = 100):
        """
        Defines an iterative training loop for a user to interact with the model.
        """
        for text_idx, text in enumerate(texts):
            if text_idx % save_after == 0 and text_idx > 0:
                self.model.save_pretrained(self.save_path())
            
            print(f"Sample {text_idx+1} of {len(texts)}\nTraining on: \n{indent(text, '|  ')}\n\n{question}")
            
            model_logits = self.predict(text).logits
            model_suggestion = model_logits.argmax().item()

            if answers:
                for idx, answer in enumerate(answers):
                    print(f"{'-> ' if idx == model_suggestion else '   '}Answer {idx}: {answer}")

            while True:
                try:
                    user_input = input("\nYour choice (press enter to skip confirm model choice): ")
                    if user_input:
                        user_choice = int(user_input)
                    else:
                        user_choice = model_suggestion
                    if user_choice < 0 or user_choice >= self.num_labels:
                        raise ValueError
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            self.backward(model_logits, user_choice)

            print("==========================================\n")

if __name__ == "__main__":
    import random

    classifier = Classifier.load_latest()

    data = []  # Replace with your data
    random.shuffle(data)

    classifier.training_loop(data, answers=["Yes", "No"], question="Is it a yes/no question?")
