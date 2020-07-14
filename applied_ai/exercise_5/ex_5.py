import warnings
import urllib.request
import re
import nltk
from rnnmorph.predictor import RNNMorphPredictor
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from tqdm import tqdm


def print_results(data_url: str, tok_freq: int, tok_gr: int):
    warnings.filterwarnings('ignore')
    nltk.download('punkt')
    nltk.download("stopwords")

    # Init predictor
    predictor = RNNMorphPredictor(language="ru")

    # Load text
    opener = urllib.request.URLopener({})
    resource = opener.open(data_url)
    charset = resource.headers.get_content_charset()
    raw_text = resource.read()

    if charset:
        raw_text = raw_text.decode(resource.headers.get_content_charset())
    else:
        raw_text = raw_text.decode('utf-8')

    # Clean text
    cleaned_text = clean_html(raw_text)

    # Tokenize sentences
    tokenized_sentences = [word_tokenize(sentence) for sentence in
                           sent_tokenize(cleaned_text)]

    # Predict
    predictions = [[pred.normal_form for pred in sent]
                   for sent in tqdm(
            predictor.predict_sentences(sentences=tokenized_sentences),
            "sentences")]
    non_uniq_tokens = [word for sentence in predictions for word in sentence]
    non_uniq_tokens = list(filter(lambda a: str.isalpha(a),
                                  non_uniq_tokens))

    print('Sentences count: ' + str(len(predictions)))
    print('Tokens count: ' + str(len(non_uniq_tokens)))

    stopwords_list = list(stopwords.words("russian"))
    fDist = FreqDist(non_uniq_tokens)
    most_common = fDist.most_common(tok_freq)
    mc_full_len = len(most_common)
    most_common = list(filter(
        lambda x: stopwords_list.count(x[0]) == 0, most_common))
    mc_reduced_len = len(most_common)
    print('Top ' + str(tok_freq) + ' words proportion: '
          + str(mc_reduced_len/mc_full_len))

    tokens_greater = list(filter(lambda x: x[1] > tok_gr, fDist.items()))
    print('Tokens used more than ' + str(tok_gr) + ' times: '
          + str(len(tokens_greater)))


def clean_html(raw_html: str):
    clean_pattern = re.compile("<.*?>")
    clean_text = re.sub(clean_pattern, " ", raw_html)
    return clean_text
