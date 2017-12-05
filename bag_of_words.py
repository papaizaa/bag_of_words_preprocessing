import os
import math
from BeautifulSoup import BeautifulSoup
from collections import Counter
from string import punctuation
from porter_stemmer import PorterStemmer


def only_letters(string):
    """Determine if a string contains only letters"""
    return all(letter.isalpha() for letter in string)


def count_words_from_files(project_path):
    """Go through each file in the project path and count the total occurance of each word, 
        the unique occurence for each document and the count of each word within each document."""
    total_count = Counter()
    individual_counts = {}
    occurence_per_doc = Counter()
    total = 0   # Total number of files in directory

    for root, dirs, files in os.walk(project_path, topdown=False):
        for name in files:
            lines = open(os.path.join(root, name), "r")
            soup = BeautifulSoup(lines)
            name = os.path.join(root, name).replace('\\', '/')[os.path.join(root, name).find('projectdata'):]
            count_dict = Counter()
            unique_occur = Counter()
            text = list((''.join(s.findAll(text=True)) for s in soup.findAll('p')))
            for y in text:
                for x in y.split():
                    key = x.rstrip(punctuation).lower()     # Remove punctuation

                    if only_letters(key) and len(key) > 0:
                        if key not in unique_occur:
                            unique_occur[key] = 1
                        count_dict[key] += 1

            individual_counts[name] = count_dict
            total_count += count_dict
            occurence_per_doc += unique_occur
            total += 1
    print "Finished Counting words in file."
    return total_count, individual_counts, occurence_per_doc, total


def calc_tf_idf(individual_counts, occurences_per_doc, total):
    """Use the algorithm tf-idf(Term Frequency - Inverted Document Frequency) to assign a weight to each word
        within a document."""
    tf_idf_res = {}
    for filename, counts in individual_counts.iteritems():
        tf_idf_file = dict()
        for key, count in individual_counts[filename].iteritems():
            if key in occurence_per_doc.keys():
                tf_idf_file[key] = tfidf(key, counts, total, occurences_per_doc)
        tf_idf_res[filename] = tf_idf_file
    print "Finished tf-idf for words in each file"
    return tf_idf_res


def remove_stop_words(total_count, individual_counts, occurence_per_doc, stops):
    """Iterate over the dictionaries and remove all stop words, words of length 1 and rare words"""
    for key in total_count.keys():
        if key in stops or len(key) == 1 or occurence_per_doc[key] <= 3:
            del total_count[key]

    for filename, counts in individual_counts.iteritems():
        for key in counts.keys():
            if key in stops or len(key) == 1 or key not in total_count.keys():
                del individual_counts[filename][key]

    for key in occurence_per_doc.keys():
        if key in stops or len(key) == 1 or key not in total_count.keys():
            del occurence_per_doc[key]

    print "Finished removing stop words."


def group_stems(total_count, individual_counts, occurence_per_doc):
    """Use the Porter Stemmer algorithm to take only the stems of words and then group them together as a single
        count. For instance, run and running might both be in the counts, hence we reduce this to just run."""
    stemmer = PorterStemmer()
    new_individual_counts = {}
    new_total_counts = Counter()
    new_occurences_per_doc = Counter()
    for file_name, counts in individual_counts.iteritems():
        file_counts = Counter()
        for word, count in counts.iteritems():
            word_stem = stemmer.stem(word, 0, len(word) - 1)
            file_counts[word_stem] += count
        new_individual_counts[file_name] = file_counts

    for word, count in total_count.iteritems():
        word_stem = stemmer.stem(word, 0, len(word) - 1)
        new_total_counts[word_stem] += count

    for word, count in occurence_per_doc.iteritems():
        word_stem = stemmer.stem(word, 0, len(word) -1)
        new_occurences_per_doc[word_stem] += count

    print "Finished grouping words by their stems."

    return new_total_counts, new_individual_counts, new_occurences_per_doc


def create_arff_file(occurence_per_doc, individual_counts, output):
    """Go through each individual count and write the arff file using the tf-idf values and the most common words as 
        features."""
    file_arff = open(output, "w")
    file_arff.write("@relation bag_of_words_train\n")
    file_arff.write("@attribute filename string\n")

    # Use the most common words in occurence per doc as your feature set
    n_most_total_counts = occurence_per_doc.most_common(1000)

    for word, count in n_most_total_counts:

        file_arff.write("@attribute {} numeric\n".format(word))

    file_arff.write("@attribute classification_tag {faculty, course, student}\n")
    file_arff.write("@data\n")

    for filename, counts in individual_counts.iteritems():

        string = filename + ", "
        for word, count in n_most_total_counts:

            if word in counts.keys():
                val = counts[word]
                string += "{:.6f}, ".format(val)
            else:
                string += "0.0, "

        string += find_label(filename) + "\n"
        file_arff.write(string)

    file_arff.close()
    print "Finished creating arff file.\n"


def find_label(root):
    if 'faculty' in root:
        return 'faculty'
    elif 'course' in root:
        return 'course'
    else:
        return 'student'


def tf(word, counter):
    return float(counter[word]) / len(counter.keys())


def idf(word, total, occurences_per_doc):
    return math.log(float(total) / occurences_per_doc[word])


def tfidf(word, counter, total_count, occurences_per_doc):
    return tf(word, counter) * idf(word, total_count, occurences_per_doc)


def get_stop_words(file_name):
    """Retrieve all the stop words from the filename"""
    lines = open(file_name, "r")
    list_lines = []
    for line in lines:
        list_lines.append(line.rstrip())
    lines.close()
    return list_lines


if __name__ == "__main__":
    stops = get_stop_words('stop_words.lst')

    print "Creating Train Arff File\n"

    total_count, individual_counts, occurence_per_doc, total = count_words_from_files('./projectdata/train/')
    total_count, individual_counts, occurence_per_doc = group_stems(total_count, individual_counts, occurence_per_doc)

    tf_idf_res = calc_tf_idf(individual_counts, occurence_per_doc, total)

    remove_stop_words(total_count, tf_idf_res, occurence_per_doc, stops)
    create_arff_file(occurence_per_doc, tf_idf_res, 'bag_of_words_train.arff')

    print "Creating Test Arff File\n"
    test_total_count, test_individual_counts, test_occurence_per_doc, test_total = \
        count_words_from_files('./projectdata/test/')
    test_total_count, test_individual_counts, test_occurence_per_doc = \
        group_stems(test_total_count, test_individual_counts, test_occurence_per_doc)

    test_tf_idf_res = calc_tf_idf(test_individual_counts, test_occurence_per_doc, test_total)

    remove_stop_words(test_total_count, test_tf_idf_res, test_occurence_per_doc, stops)
    # Make sure to give the occurence_per_doc from the training set
    create_arff_file(occurence_per_doc, test_tf_idf_res, 'bag_of_words_test.arff')


