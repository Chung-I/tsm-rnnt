from typing import Dict, List
import itertools
import warnings

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length, START_SYMBOL, END_SYMBOL
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.tokenizer import Tokenizer


@TokenIndexer.register("just-characters")
class CharactersIndexer(TokenIndexer[List[int]]):
    """
    This :class:`TokenIndexer` represents tokens as lists of character indices.

    Parameters
    ----------
    namespace : ``str``, optional (default=``token_characters``)
        We will use this namespace in the :class:`Vocabulary` to map the characters in each token
        to indices.
    character_tokenizer : ``CharacterTokenizer``, optional (default=``CharacterTokenizer()``)
        We use a :class:`CharacterTokenizer` to handle splitting tokens into characters, as it has
        options for byte encoding and other things.  The default here is to instantiate a
        ``CharacterTokenizer`` with its default parameters, which uses unicode characters and
        retains casing.
    start_tokens : ``List[str]``, optional (default=``None``)
        These are prepended to the tokens provided to ``tokens_to_indices``.
    end_tokens : ``List[str]``, optional (default=``None``)
        These are appended to the tokens provided to ``tokens_to_indices``.
    min_padding_length: ``int``, optional (default=``0``)
        We use this value as the minimum length of padding. Usually used with :class:``CnnEncoder``, its
        value should be set to the maximum value of ``ngram_filter_sizes`` correspondingly.
    token_min_padding_length : ``int``, optional (default=``0``)
        See :class:`TokenIndexer`.
    """
    # pylint: disable=no-self-use
    def __init__(self,
                 namespace: str = 'characters',
                 character_tokenizer: Tokenizer = CharacterTokenizer(),
                 start_tokens: List[str] = None,
                 end_tokens: List[str] = None,
                 min_padding_length: int = 0,
                 lowercase_tokens: bool = False,
                 token_min_padding_length: int = 0) -> None:
        super().__init__(token_min_padding_length)
        if min_padding_length == 0:
            url = "https://github.com/allenai/allennlp/issues/1954"
            warnings.warn("You are using the default value (0) of `min_padding_length`, "
                          f"which can cause some subtle bugs (more info see {url}). "
                          "Strongly recommend to set a value, usually the maximum size "
                          "of the convolutional layer size when using CnnEncoder.",
                          UserWarning)
        self._min_padding_length = min_padding_length
        self._namespace = namespace
        self._character_tokenizer = character_tokenizer
        self.lowercase_tokens = lowercase_tokens

        self._start_tokens = [Token(st) for st in (start_tokens or [])]
        self._end_tokens = [Token(et) for et in (end_tokens or [])]

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        if token.text is None:
            raise ConfigurationError('CharactersIndexer needs a tokenizer that retains text')
        for character in self._character_tokenizer.tokenize(token.text):
            # If `text_id` is set on the character token (e.g., if we're using byte encoding), we
            # will not be using the vocab for this character.
            text = character.text
            if getattr(character, 'text_id', None) is None:
                if self.lowercase_tokens:
                    text = text.lower()
                counter[self._namespace][text] += 1

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[List[int]]]:
        indices: List[int] = []
        for token in itertools.chain(self._start_tokens, tokens, self._end_tokens):
            token_indices: List[int] = []
            if token.text is None:
                raise ConfigurationError('CharactersIndexer needs a tokenizer that retains text')
            if token.text in [START_SYMBOL, END_SYMBOL]:
                token_indices = [vocabulary.get_token_index(token.text, self._namespace)]
            else:
                for character in self._character_tokenizer.tokenize(token.text):
                    if getattr(character, 'text_id', None) is not None:
                        # `text_id` being set on the token means that we aren't using the vocab, we just
                        # use this id instead.
                        index = character.text_id
                    else:
                        text = character.text
                        if self.lowercase_tokens:
                            text = text.lower()
                        index = vocabulary.get_token_index(text, self._namespace)
                    token_indices.append(index)
            indices += token_indices
        return {index_name: indices}

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:
        return {}

    @overrides
    def get_padding_token(self) -> List[int]:
        return []

    @overrides
    def as_padded_tensor(self,
                         tokens: Dict[str, List[int]],
                         desired_num_tokens: Dict[str, int],
                         padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
        return {key: torch.LongTensor(pad_sequence_to_length(val, desired_num_tokens[key]))
                for key, val in tokens.items()}
