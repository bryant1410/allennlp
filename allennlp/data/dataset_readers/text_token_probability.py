import copy
import logging
from typing import Dict

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import IndexField, ListField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import Token, WordTokenizer
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

logger = logging.getLogger(__name__)


@DatasetReader.register("text_token_probability")
class TextTokenProbabilityDatasetReader(DatasetReader):
    """
    Reads a text file and converts it into a ``Dataset`` suitable for computing the probabilities
    of each word token, given the rest of the text, for a masked language model.

    The given text can't contain the [MASK] token.
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 mask_special_tokens: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._mask_special_tokens = mask_special_tokens

    @overrides
    def _read(self, file_path: str):
        with open(cached_path(file_path)) as text_file:
            for sentence in text_file:
                yield self.text_to_instance(sentence)

    @overrides
    def text_to_instance(self, sentence: str) -> Instance:
        tokens = self._tokenizer.tokenize(sentence)

        assert all(token.text != '[MASK]' for token in tokens)

        text_fields = []
        mask_position_fields = []

        for i in range(len(tokens)):
            if not self._mask_special_tokens and tokens[i].text.startswith('[') and tokens[i].text.endswith(']'):
                continue

            tokens_copy = copy.deepcopy(tokens)
            tokens_copy[i] = Token('[MASK]')

            text_field = TextField(tokens_copy, self._token_indexers)
            text_fields.append(text_field)

            mask_position_fields.append(ListField([IndexField(i, text_field)]))

        # TODO: I think there's a problem if the masked tokens get split into multiple word pieces...

        return Instance({
            'masked_sentences_tokens': ListField(text_fields),
            'mask_positions': ListField(mask_position_fields),
            'unmasked_sentence_tokens': TextField(tokens, self._token_indexers)
        })
