import pytest

from src.model.tokenizer import CharacterTokenizer


def test_encode_text():
    tokenizer = CharacterTokenizer()

    token_ids = tokenizer.encode("Hi")

    assert token_ids == [41, 74]


def test_decode_text():
    tokenizer = CharacterTokenizer()

    text = tokenizer.decode([41, 74])

    assert text == "Hi"


def test_encode_and_decode():
    tokenizer = CharacterTokenizer()

    original_text = "Hello World!"

    token_ids = tokenizer.encode(original_text)
    decoded_text = tokenizer.decode(token_ids)

    assert decoded_text == original_text


def test_unknown_character():
    tokenizer = CharacterTokenizer()

    token_ids = tokenizer.encode("Hello\n")

    assert token_ids[-1] == tokenizer.unknown_token


def test_beginning_and_end_tokens():
    tokenizer = CharacterTokenizer()

    token_ids = tokenizer.encode(
        "Hi",
        add_bos=True,
        add_eos=True
    )

    assert token_ids[0] == tokenizer.bos_token
    assert token_ids[-1] == tokenizer.eos_token


def test_decode_ignores_special_tokens():
    tokenizer = CharacterTokenizer()

    text = tokenizer.decode([96, 41, 74, 97, 98])

    assert text == "Hi"


def test_invalid_token_id():
    tokenizer = CharacterTokenizer()

    with pytest.raises(ValueError):
        tokenizer.decode([200])


def test_invalid_vocabulary_size():
    with pytest.raises(ValueError):
        CharacterTokenizer(vocab_size=50)