import nltk
from nltk import ConditionalFreqDist
from nltk.corpus.reader import TaggedCorpusReader
from nltk.util import bigrams
from pathlib import Path



def read_corpus():
    return TaggedCorpusReader(root= str(Path(__file__).parent),
                              fileids='MIM-GOLD.sent',
                              sep='/')

def run_analysis():
    result = {}

    doc_reader = read_corpus()
    sentences = doc_reader.sents()
    individual_sentence = [word for word in sentences[99]]

    tokens = doc_reader.words()
    token_mostcommon = [token + " => " + str(times) for (token, times) in nltk.FreqDist(tokens).most_common(10)]

    tags = [tag for (_, tag) in doc_reader.tagged_words()]
    tag_mostcommon = [tag + " => " + str(times) for (tag, times) in nltk.FreqDist(tags).most_common(20)]

    cdfr = ConditionalFreqDist()
    
    tags_bigrams = list(bigrams(tags))

    for tag1, tag2 in tags_bigrams:
        cdfr[tag1][tag2] += 1
    following_tags_mostcommon = [tag + " => " + str(freq) for (tag, freq) in cdfr['AF'].most_common(10)]

    result['Number of sentences'] = len(sentences)
    result['Sentence no. 100'] = "\n" + ' '.join(individual_sentence) + "\n"
    result['Number of tokens'] = len(tokens)
    result['Number of type'] = str(len(set(tokens))) +  '\n'
    result['The 10 most frequent tokens'] = "\n" + '\n'.join(token_mostcommon) + '\n'
    result['The 20 most frequent tags'] = "\n" + '\n'.join(tag_mostcommon) + '\n'
    result['The 10 most frequent PoS tags following the tag ’af’'] = "\n" + '\n'.join(following_tags_mostcommon) + '\n'

    return result


def process_result(result):
    return "\n".join(f"{key}: {value}" for key, value in result.items())

def write_result(result):
    print("Saving results...")
    with open("mimGoldAnalysisResult.txt", "w", encoding="utf-8") as file:
        file.write(result)
    print("Saving completed successfully!")


def main():
    result = run_analysis()
    text_result = process_result(result)
    print(f'[ANALYSIS RESULT] \n{text_result}')
    write_result(text_result)
    print("The results are available at mimGoldAnalysisResult.txt")

if __name__ == '__main__':
    main()