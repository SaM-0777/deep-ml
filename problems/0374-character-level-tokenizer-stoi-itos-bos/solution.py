class CharTokenizer:
    def __init__(self, text: str):
        """
        Build a character-level tokenizer from the input text.
        
        Args:
            text: A string used to build the vocabulary.
        """
        # Your code here
        chars = sorted(set(text))

        self.stoi = {'<BOS>': 0, '<EOS>': 1}
        self.stoi.update({char: i + 2 for i, char in enumerate(chars)})

        self.itos = {i: token for token, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, text: str) -> list:
        """
        Encode a string into a list of token indices.
        
        Args:
            text: The string to encode.
        Returns:
            List of integer indices.
        """
        # Your code here
        return [self.stoi['<BOS>']] + [self.stoi[char] for char in text] + [self.stoi['<EOS>']]

    def decode(self, indices: list) -> str:
        """
        Decode a list of token indices back into a string.
        
        Args:
            indices: List of integer indices.
        Returns:
            Decoded string.
        """
        # Your code here
        return ''.join(self.itos[i] for i in indices)