from feature import load_set, extract_features, load_clean_descriptions, load_photo_features
from keras.models import load_model
from pickle import load
from evaluate import evaluate_model
from feature import
from generate import generate_desc
# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load test set
filename = 'Flicker8k_text/Flickr_8k.testImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features.pkl', test)
print('Photos: test=%d' % len(test_features))

# load the model
filename = 'model/model_19.h5'
model = load_model(filename=filename)
# evaluate model
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

# load and prepare the photograph
photo = extract_features('example.jpg')
# generate description
description = generate_desc(model, tokenizer, photo, max_length)
print(description)
