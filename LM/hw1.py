import argparse
import math
import random
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List
from typing import Tuple
from typing import Generator


# Generator for all n-grams in text
# n is a (non-negative) int
# text is a list of strings
# Yields n-gram tuples of the form (string, context), where context is a tuple of strings
def get_ngrams(n: int, text: List[str]) -> Generator[Tuple[str, Tuple[str, ...]], None, None]:
    # Pad the text with start tokens '<s>' and the end token '</s>'
    padded_text = ['<s>'] * (n - 1) + text + ['</s>']
    
    # Iterate through the text to yield n-grams
    for i in range(n - 1, len(padded_text)):
        word = padded_text[i]
        context = tuple(padded_text[i - n + 1:i])
        yield (word, context)


# Loads and tokenizes a corpus
# corpus_path is a string
# Returns a list of sentences, where each sentence is a list of strings
def load_corpus(corpus_path: str) -> List[List[str]]:
    import nltk
    nltk.download('punkt')

    # Initialize a list to store the sentences
    sentences = []
    
    # Open the corpus file
    with open(corpus_path, 'r', encoding='utf-8') as file:
        # Split the text into paragraphs
        paragraphs = file.read().split('\n\n')

        for paragraph in paragraphs:
            if paragraph:
                # Use NLTK's sentence tokenizer to split the paragraph into sentences
                paragraph_sentences = sent_tokenize(paragraph)
                
                for sentence in paragraph_sentences:
                    # Tokenize each sentence into words
                    words = word_tokenize(sentence)
                    sentences.append(words)
    
    return sentences




# Builds an n-gram model from a corpus
# n is a (non-negative) int
# corpus_path is a string
# Returns an NGramLM
def create_ngram_lm(n: int, corpus_path: str) -> 'NGramLM':
    # Load the data from the corpus path
    corpus_sentences = load_corpus(corpus_path)
    
    # Create a new NGramLM with the specified n-gram size
    ngram_model = NGramLM(n)
    
    # Update the NGramLM with each sentence from the loaded corpus
    for sentence in corpus_sentences:
        ngram_model.update(sentence)
    
    # Return the trained NGramLM
    return ngram_model


# An n-gram language model
class NGramLM:
    def __init__(self, n: int):
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocabulary = set()

    # Updates internal counts based on the n-grams in text
    # text is a list of strings
    # No return value
    def update(self, text: List[str]) -> None:
        # Use the get_ngrams function to get n-grams of the appropriate size
        ngrams = get_ngrams(self.n, text)
        
        # Iterate through the n-grams and update counts
        for word, context in ngrams:
            # Update n-gram counts
            if (word, context) in self.ngram_counts:
                self.ngram_counts[(word, context)] += 1
            else:
                self.ngram_counts[(word, context)] = 1
            
            # Update context counts
            if context in self.context_counts:
                self.context_counts[context] += 1
            else:
                self.context_counts[context] = 1

            # Update Vocabulary
            if not word in self.vocabulary:
                self.vocabulary.add(word)


    # Calculates the MLE probability of an n-gram
    # word is a string
    # context is a tuple of strings
    # delta is an float
    # Returns a float
    def get_ngram_prob(self, word: str, context: Tuple[str, ...], delta= .0) -> float:
        # Calculate the probability of the word given the context using MLE
        ngram = (word, context)
        context_count = self.context_counts.get(context, 0)
        vocabulary_size = len(self.vocabulary)
        
        if context_count == 0:
            # Unseen context: Apply add-one (Laplace) smoothing
            return 1 / (vocabulary_size)
        
        ngram_count = self.ngram_counts.get(ngram, 0)
        return (ngram_count + delta) / (context_count + delta * vocabulary_size) if ngram_count > 0 else 1 / (vocabulary_size)        


    # Calculates the log probability of a sentence
    # sent is a list of strings
    # delta is a float
    # Returns a float
    def get_sent_log_prob(self, sent: List[str], delta=.0) -> float:
        ngrams = get_ngrams(self.n, sent)
        log_prob_sum = 0.0

        for word, context in ngrams:
            # Calculate the logarithm of the n-gram probability using get_ngram_prob
            log_prob = math.log(self.get_ngram_prob(word, context, delta), 2)
            log_prob_sum += log_prob

        return log_prob_sum


    # Calculates the perplexity of a language model on a test corpus
    # corpus is a list of lists of strings
    # delta is a float
    # Returns a float
    def get_perplexity(self, corpus: List[List[str]], delta=0.) -> float:
        total_log_prob = 0.0
        total_tokens = 0

        for sentence in corpus:
            log_prob = self.get_sent_log_prob(sentence)
            total_log_prob += log_prob
            total_tokens += len(sentence)

        average_log_prob = total_log_prob / total_tokens

        # Calculate perplexity using math.pow
        perplexity = math.pow(2, -average_log_prob)

        return perplexity

    # Samples a word from the probability distribution for a given context
    # context is a tuple of strings
    # delta is an float
    # Returns a string
    def generate_random_word(self, context: Tuple[str, ...], delta=.0) -> str:
        # Sort the vocabulary alphabetically
        sorted_vocabulary = sorted(self.vocabulary)

        # Generate a random number r in the range [0.0, 1.0)
        r = random.random()

        # Initialize the probability range start and end
        range_start = 0.0

        for word in sorted_vocabulary:
            # Calculate the n-gram probability for the word given the context
            prob = self.get_ngram_prob(word, context, delta)

            # Update the range end
            range_end = range_start + prob

            # Check if the random number falls within this word's range
            if range_start <= r < range_end:
                return word  # Return the word corresponding to the selected range

            # Move the range start to the end of the current word's range
            range_start = range_end

        # If no word is selected, return None (this should not happen under normal circumstances)
        return None

    # Generates a random sentence
    # max_length is an int
    # delta is a float
    # Returns a string
    def generate_random_text(self, max_length: int, delta=.0) -> str:
        generated_text = []
        current_context = ('<s>',) * (self.n - 1)  # Initialize context with start tokens

        for _ in range(max_length):
            # Generate a random word using the current context
            word = self.generate_random_word(current_context, delta)

            # Stop if the stop token '</s>' is generated
            if word == '</s>':
                break

            # Append the generated word to the text and update the context
            generated_text.append(word)
            current_context = current_context[1:] + (word,)

        # Join the generated words to form a single string
        generated_sentence = ' '.join(generated_text)
        return generated_sentence


def main(corpus_path: str, delta: float, seed: int):
    trigram_lm = create_ngram_lm(3, corpus_path)
    s1 = 'God has given it to me, let him who touches it beware!'
    s2 = 'Where is the prince, my Dauphin?'

    print(trigram_lm.get_sent_log_prob(word_tokenize(s1), 0.5))
    print(trigram_lm.get_sent_log_prob(word_tokenize(s2), 0.5))


    new_lm = create_ngram_lm(5, corpus_path)
    print(new_lm.generate_random_text(10, 0.1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS6320 HW1")
    # parser.add_argument('corpus_path', nargs="?", type=str, default='warpeace.txt', help='Path to corpus file')
    parser.add_argument('corpus_path', nargs="?", type=str, default='shakespeare.txt', help='Path to corpus file')
    parser.add_argument('delta', nargs="?", type=float, default=.0, help='Delta value used for smoothing')
    parser.add_argument('seed', nargs="?", type=int, default=82761904, help='Random seed used for text generation')
    args = parser.parse_args()
    random.seed(args.seed)
    main(args.corpus_path, args.delta, args.seed)
