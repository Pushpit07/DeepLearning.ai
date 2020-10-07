import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
	'I have a dog',
	'I have a cat',
	'You have a dog!',
	'Do you think that a dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100, oov_token = "<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

test_data = [
	'I really love a cat',
	'My dog loves my Manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)