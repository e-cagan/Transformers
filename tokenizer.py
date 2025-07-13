from datasets import load_dataset
import json

# Load the dataset
ds = load_dataset("bentrevett/multi30k")

class Tokenizer:
    """Class for tokenizer."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.vocab = dict()
        self.id = list()
        self.word = list()

    def build_vocab(self):
        """Function for building a vocabulary for tokenizer."""

        # Take the english sentences
        english_sentences = [item["en"] for item in self.dataset["train"]]

        # Create the set of words
        words = set()
        for sentence in english_sentences:
            for word in sentence.lower().strip().split():
                words.add(word)

        # Special tokens
        all_words = ["<pad>", "<sos>", "<eos>", "<unk>"] + sorted(list(words))

        self.word = all_words
        self.id = list(range(len(all_words)))
        self.vocab = {word: idx for idx, word in enumerate(all_words)}

        return self.vocab

    def encode(self, text):
        """Function for encoding the text."""

        # Create tokens by lowering the text and splitting word-based
        tokens = text.lower().split()
        tokens = ["<sos>", *tokens, "<eos>"]

        # Create a list for token ids
        token_ids = list()

        # Check words
        for word in tokens:
            if word in self.vocab:
                token_ids.append(self.vocab[word])
            else:
                token_ids.append(self.vocab["<unk>"])

        return token_ids

    def decode(self, token_ids):
        """Function for decoding the text."""

        # Create a decoded tokens list
        decoded_tokens = list()

        # Create a id 2 word dictionary
        id2word = {idx: word for idx, word in enumerate(self.word)}

        # Check ids
        for id in token_ids:
            if id in id2word:
                decoded_tokens.append(id2word[id])
            else:
                decoded_tokens.append("<unk>")

        return " ".join([tok for tok in decoded_tokens if tok not in ["<sos>", "<eos>", "<pad>"]])

    def save_vocab(self, filepath):
        """Function for saving vocabulary data."""

        # Create data
        data = {
        "vocab": self.vocab,
        "word": self.word
        }
        
        # Open the file as write mode and write the data on the file
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load_vocab(self, filepath):
        """Function for loading vocabulary data"""

        # Open the file as read mode
        with open(filepath, "r") as f:
            data = json.load(f)

        # Load the vocab and word
        self.vocab = data["vocab"]
        self.word = data["word"]
        self.id = list(range(len(self.word)))

def main():
    tokenizer = Tokenizer(ds)
    tokenizer.build_vocab()
    tokenizer.save_vocab("data/vocab.json")
    tokenizer.load_vocab("data/vocab.json")
    print(tokenizer.encode("how are you"))
    print(tokenizer.decode(tokenizer.encode("how are you")))

if __name__ == '__main__':
    main()