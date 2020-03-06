def pad(token_list, target_len):

    end = "</s>"
    new_list = token_list.copy()

    while len(new_list) < target_len:
        new_list.append(end)

    return token_list + [end] * (target_len - len(token_list))


def create_ngrams(token_list, n):
    """
    this function takes in a list of tokens and forms a list of n-grams (tuples)

    INPUT:
    token_list - a list of tokens to be converted into n-grams
    n - the length of a token sequence in an n-gram
    OUTPUT:
    n_grams - a list of n-gram tuples
    """
    if n > len(token_list):
        print("The N is too large.")

    n_grams = []
    for i in range(len(token_list) - n + 1):
        n_grams.append(tuple(token_list[i:i + n]))

    return n_grams


def get_counts(sentence_list, n):
    """
    this function takes in a list of tokenized and padded sentences,
    forms a list of n-grams and gives out a dictionary
    with counts for every seen n-gram

    INPUT:
    sentence_list - a list of tokenized and padded sentences to be converted into n-grams
    n - the length of a token sequence
    OUTPUT:
    n_gram_dict - a dictionary of n_gram history parts as keys,
    where their values are a dictionary of all continuations and their counts
    {('a',): {'b': 3 'c': 4}

    """
    n_gram_dict = {}
    # ngrams = make_n_grams(sentence_list, 2)

    # print(sentence_list)
    # print()

    for sentence in sentence_list:
        ngrams = create_ngrams(sentence, n - 1)

        j = n - 1
        for index in range(0, len(ngrams)):
            pair = ngrams[index]
            if j < len(sentence):
                next_word = sentence[j]

                if pair in n_gram_dict:

                    if next_word in n_gram_dict[pair]:
                        n_gram_dict[pair][next_word] += 1
                    else:
                        n_gram_dict[pair][next_word] = 1

                else:
                    n_gram_dict[pair] = {
                        next_word: 1
                    }

            j += 1

    # print(n_gram_dict)

    return n_gram_dict
