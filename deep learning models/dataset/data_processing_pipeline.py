import numpy as np
import pandas as pd
# from LSTM_baseline.data_rough_processing import get_processed_df
import string
import re
from emot.emo_unicode import UNICODE_EMOJI

MICROTEXT_MAP = {
    "lol": "laugh out loud",
    "lmao": "laugh my ass off",
    "cuz": "because",
    "bc": "because",
    "bcuz": "because",
    "bcoz": "because",
    "b/c": "because",
    "btw": "by the way",
    "omg": "oh my god",
    "omfg": "oh my fucking god",
    "wtf": "what the fuck",
    "tf": "the fuck",
    "convo": "conversation",
    "ngl": "not gonna lie",
    "tbh": "to be honest",
    "js": "just saying",
    "xd": "laugh",
}

STOP_WORD = {
    None
}


# %% wrapper
# the pipeline is a collection of functions that can be applied to a dataframe
class DataProcessingPipeline:
    def __init__(self, callbacks: list):
        self.callbacks = callbacks

    def __call__(self, text):
        for callback in self.callbacks:
            text = callback(text)
        return text


# %% callbacks

def case_folding(text: str) -> str:
    return text.lower()


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))


def replace_emoji_with_text(text: str) -> str:
    for emot in UNICODE_EMOJI:
        text = text.replace(emot, " " + "_".join(UNICODE_EMOJI[emot].replace(",", "").replace(":", "").split()) + " ")
    return text


def remove_emoji(text: str) -> str:
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
    pass


class ReplaceWords:
    """
    Repalaes words in a text with a given map, by default it uses
     our predefined microtext map
    """

    def __init__(self, word_map=None):
        if word_map is None:
            word_map = MICROTEXT_MAP
        self.word_map = word_map

    def __call__(self, text: str) -> str:
        for word in self.word_map:
            text = re.sub(fr"\b{word}\b", self.word_map[word], text, flags=re.IGNORECASE)
        return text


# if the function is stateful, e.g. relies a list of stop words, then it should be a class
class RemoveStopWords:
    def __init__(self, stop_words: list):
        self.stop_words = stop_words

    def __call__(self, text):
        return " ".join([word for word in text.split() if word not in self.stop_words])


# if the function is stateful, e.g. relies a list of stop words, then it should be a class
# class StatefulFunc:
#     def __init__(self, state):
#         self.state = state
#
#     def __call__(self, text):
#         return text


# %%
# df = get_processed_df()
#
# pipeline = DataProcessingPipeline([
#     case_folding,
#     replace_emoji_with_text,
# ])
#
#
# temps = df["Text"].apply(pipeline)


if __name__ == '__main__':
    a = ReplaceWords(MICROTEXT_MAP)
    print(a(" xd "))
