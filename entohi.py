import numpy as np
import re
import pickle
import os
import seaborn as sns
import string
import IPython
from gtts import gTTS
from transformers import MarianMTModel, MarianTokenizer

# Load the pre-trained model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-hi'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def convert(article_en):
  input_ids = tokenizer.encode(article_en, return_tensors="pt")
  output = model.generate(input_ids)
  translated_text = tokenizer.decode(output[0], skip_special_tokens=True)
  return translated_text
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

