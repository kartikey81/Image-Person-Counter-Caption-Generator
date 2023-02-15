import numpy as np
import re
import pickle
import os
import seaborn as sns
import string
import IPython
from gtts import gTTS
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")
def convert(article_en):
  model_inputs = tokenizer(article_en, return_tensors="pt")
  generated_tokens = model.generate(
    **model_inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
      )
  translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
  return translation[0]
def hindispeech(text):
  language = 'hi' #hindi
  speech = gTTS(text = text, lang = language, slow = False)
  speech.save('medium_hindi_2.wav')
  w='medium_hindi_2.wav'
  return w
def englishspeech(text):
  language = 'en' #hindi
  speech = gTTS(text = text, lang = language, slow = False)
  speech.save('medium_english_2.wav')
  w='medium_english_2.wav'
  return w
#to play string in wav

