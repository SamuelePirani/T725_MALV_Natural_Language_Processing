import nltk
from nltk.corpus import treebank

nltk.download('averaged_perceptron_tagger_eng')

taggers = {"Affix tagger": nltk.AffixTagger,
           "Unigram tagger": nltk.UnigramTagger ,
           "Bigram tagger": nltk.BigramTagger ,
           "Trigram tagger": nltk.TrigramTagger}


def prepare_sets():
    training_set = treebank.tagged_sents()[:3500]
    test_set = treebank.tagged_sents()[3500:]
    return training_set, test_set

def print_results(value_description, value, break_results_line=False, new_line=False):
    if break_results_line:
        print(value_description + ":\n" + str(value))
    else:
        print(value_description + ": " + str(value))
    if new_line:
        print("\n")

def untag_set(training_set):
    sentences_to_tag = []
    for sentences_tagged in training_set:
        sentece_untagged = [word_tagged[0] for word_tagged in sentences_tagged]
        sentences_to_tag.append(sentece_untagged)
    return sentences_to_tag

def evaluate_custom(predicted_tag, actual_tag):
    correct = 0
    total = 0
    for i in range(len(predicted_tag)):
        for j in range(len(predicted_tag[i])):
            if predicted_tag[i][j][1] == actual_tag[i][j][1]:
                correct += 1
            total += 1
    return (correct / total).__round__(3) * 100

def build_model(model, train_set, backoff=None):
    return model(train_set, backoff=backoff)

def run_evaluation(model, test_set):
    return model.accuracy(test_set).__round__(3) * 100


def main():
    training_set, test_set = prepare_sets()

    print_results("Number of training sentences", len(training_set))
    print_results("Number of test sentences", len(test_set), new_line=True)
    print_results("First sentence in test corpus", test_set[0], break_results_line=True ,new_line=True)

    print_results("Tagging accuracies", "-----------------", break_results_line=True)


    # Without backoff, the Bigram Tagger assigns tags based on the current word and the previous tag.
    # If a word-tag pair (bigram) isn’t found in the training data, it fails to tag, resulting in low accuracy (13.5%).
    for key, value in taggers.items():
        model = build_model(value, training_set)
        print_results(key, str(run_evaluation(model, test_set)) + "%")
    print()


    #Backoff solves this by allowing the tagger to fall back to simpler models like the Unigram or Affix Tagger
    #when it encounters unknown contexts. This layered fallback greatly improves tagging coverage and boosts
    #accuracy to 90.8%.
    print_results("Tagging accuracies with backoff", "-------------------------", break_results_line=True)

    current_model = None

    for i, (key, value) in enumerate(taggers.items()):
        if i == 0:
            current_model = build_model(value, training_set)
            print_results(key, str(run_evaluation(current_model, test_set)) + "%")
        else:
            model = build_model(value, training_set,  current_model)
            print_results(key, str(run_evaluation(model, test_set)) + "%")
            current_model = model
    print()

    default_tagger = nltk.pos_tag_sents(untag_set(test_set))
    print_results("Accuracy of the default tagger in NLTK",
                  str(evaluate_custom(default_tagger, test_set)) + "%")


if __name__ == '__main__':
    main()


