import sys

import nltk, re
from nltk.corpus import gutenberg

nltk.download('gutenberg')

def run_corpus_analysis(corpus_name):

    result = {'Text': corpus_name}
    stopword = nltk.corpus.stopwords.words('english')

    corpus_token = gutenberg.words(corpus_name)
    corpus_types = set(corpus_token)
    corpus_types_refined = [word for word in corpus_types if word not in stopword]

    corpus_tokenfreq = nltk.FreqDist(corpus_token)
    corpus_token_mostcommon = corpus_tokenfreq.most_common(10)

    corpus_longtypes = [word for word in corpus_types if len(word) > 13]

    corpus_matchexp = set(re.findall(r'\b\w*ation\b', gutenberg.raw(corpus_name)))

    result['Tokens'] = len(corpus_token)
    result['Types'] = len(corpus_types)
    result['Types excluding stop words'] = len(corpus_types_refined)
    result['10 most common tokens'] = corpus_token_mostcommon
    result['Long Types'] = corpus_longtypes
    result['Nouns ending in ’ation’'] = corpus_matchexp

    return result

def process_result(result):
    return "\n".join(f"{key}: {value}" for key, value in result.items())

def write_result(result):
    print("Saving results...")
    with open("analysisResult.txt", "w", encoding="utf-8") as file:
        file.write(result)
    print("Saving completed successfully!")


def main():
    result = run_corpus_analysis("austen-emma.txt")
    final_result = process_result(result)
    print(f'[ANALYSIS RESULT] \n{final_result}')
    write_result(final_result)
    print("The results are available at analysisResult.txt")


if __name__ == "__main__":
    main()