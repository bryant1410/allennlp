from typing import Dict

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import LanguageModelHead, TextFieldEmbedder
from allennlp.nn import InitializerApplicator


@Model.register('text_token_probability')
class TextTokenProbabilityModel(Model):
    def __init__(self, vocab: Vocabulary, text_field_embedder: TextFieldEmbedder,
                 language_model_head: LanguageModelHead, initializer: InitializerApplicator = None) -> None:
        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder
        self._language_model_head = language_model_head
        if initializer:
            initializer(self)

    @overrides
    def forward(self, masked_sentences_tokens: Dict[str, torch.LongTensor], mask_positions: torch.LongTensor,
                unmasked_sentence_tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_sentences, 1)
        batch_size, num_sentences, num_tokens = mask_positions.size()

        assert num_tokens == 1

        # Shape: (batch_size, num_sentences, num_tokens, embedding_size)
        encoded = self._text_field_embedder(masked_sentences_tokens, num_wrapping_dims=1)

        # Does advanced indexing to get the embeddings of just the mask positions,
        # which is what we're trying to predict.
        batch_index = torch.arange(0, batch_size).long().unsqueeze(-1).unsqueeze(-1)
        sentence_index = torch.arange(0, num_sentences).unsqueeze(0).unsqueeze(-1)
        # Shape: (batch_size, num_sentences, embedding_size)
        mask_embeddings = encoded[batch_index, sentence_index, mask_positions].squeeze(2)

        # Shape: (batch_size, num_sentences, vocab_size)
        logits = self._language_model_head(mask_embeddings)

        original_token_indices = unmasked_sentence_tokens['tokens'][batch_index, sentence_index, mask_positions]\
            .squeeze(2)

        # Shape: (batch_size, num_sentences)
        logits_original_tokens = 1 - logits[batch_index, sentence_index, original_token_indices].squeeze(-1)

        return {'unlikeliness': F.softmax(logits_original_tokens, dim=-1)}
