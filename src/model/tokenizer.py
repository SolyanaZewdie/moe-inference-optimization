class CharacterTokenizer:
    def __init__(self, vocab_size=128):
        if vocab_size < 99:
            raise ValueError("Vocabulary must contain at least 99 tokens")

        self.vocab_size = vocab_size

        self.unknown_token = 0
        self.bos_token = 96
        self.eos_token = 97
        self.pad_token = 98

    def encode(self, text, add_bos=False, add_eos=False):
        token_ids = []

        if add_bos:
            token_ids.append(self.bos_token)

        for character in text:
            character_id = ord(character) - 31

            if character_id < 1 or character_id > 95:
                character_id = self.unknown_token

            token_ids.append(character_id)

        if add_eos:
            token_ids.append(self.eos_token)

        return token_ids

    def decode(self, token_ids):
        characters = []

        for token_id in token_ids:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError("Invalid token ID")

            if 1 <= token_id <= 95:
                characters.append(chr(token_id + 31))

        return "".join(characters)