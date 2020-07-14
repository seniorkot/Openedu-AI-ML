import urllib.request
import re
import nltk
from nltk.tokenize import word_tokenize
from tqdm.notebook import tqdm
from nltk import FreqDist
from nltk.corpus import stopwords


def print_results(data_url: str, tok_freq: int, tok_gr: int):
    nltk.download("wordnet")
    nltk.download('punkt')
    nltk.download("stopwords")

    # Init lemmatizer
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

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

    # Get tokens
    tokens = word_tokenize(cleaned_text)

    # Get lemmas
    lemmas = [lemmatizer.lemmatize(lemma) for lemma in tqdm(tokens)]
    lemmas = list(filter(lambda a: str.isalpha(a), lemmas))
    print('Lemmas count: ' + str(len(lemmas)))

    # Get tops
    stopwords_list = list(stopwords.words("english"))
    fDist = FreqDist(lemmas)
    most_common = fDist.most_common(tok_freq)
    mc_full_len = len(most_common)
    most_common = list(filter(
        lambda x: stopwords_list.count(x[0]) == 0, most_common))
    mc_reduced_len = len(most_common)
    print('Top ' + str(tok_freq) + ' words proportion: '
          + str(mc_reduced_len / mc_full_len))

    tokens_greater = list(filter(lambda x: x[1] < tok_gr, fDist.items()))
    print('Tokens used more than ' + str(tok_gr) + ' times: '
          + str(len(tokens_greater)))


def clean_html(raw_text: str):
    clean_pattern = re.compile("End of the Project Gutenberg EBook.*")
    clean_text = re.sub(clean_pattern, "",
                        raw_text.replace("\n", " ").replace("\r", " "))
    return clean_text
