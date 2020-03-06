
data_file = "testdata.txt"
ngram_size = 6


def read_file(filename):
    with open(filename, 'r') as file:
        lines = list(map(lambda line: line.strip(), file.readlines()))
    return lines


def create_unique_words(lines):
    unique_words = []
    for sentence in lines:
        for word in sentence.split():
            if word not in unique_words:
                unique_words.append(word)

        unique_words = sorted(unique_words)

    unique_words.append('</s>')
    nof_unique_words = len(unique_words)

    return unique_words, nof_unique_words


def build_index(unique_words):
    d = {}
    for i, word in enumerate(unique_words):
        d[word] = i

    return d

lines = read_file(data_file)
unique_words, nof_unique_words = create_unique_words(lines)


print(unique_words, "\n", nof_unique_words)
