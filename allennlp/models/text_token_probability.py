from typing import Dict

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import LanguageModelHead, TextFieldEmbedder
from allennlp.nn import InitializerApplicator


@Model.register("text_token_probability")
class TextTokenProbabilityModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        language_model_head: LanguageModelHead,
        initializer: InitializerApplicator = None,
    ) -> None:
        super().__init__(vocab)
        self._text_field_embedder = text_field_embedder
        self._language_model_head = language_model_head
        if initializer:
            initializer(self)

    @overrides
    def forward(
        self,
        masked_sentences_tokens: Dict[str, torch.LongTensor],
        mask_positions: torch.LongTensor,
        unmasked_sentence_tokens: Dict[str, torch.LongTensor],
    ) -> Dict[str, torch.Tensor]:
        # mask_positions shape: (batch_size, num_sentences, 1, 1)
        # IndexField provides and extra 1.
        # The previous 1 is because there should only be 1 masked token per sentence.
        mask_positions = mask_positions.squeeze(-1).squeeze(-1)

        # Shape: (batch_size, num_sentences)
        batch_size, num_sentences = mask_positions.size()

        # There's one masked token per sentence, so `num_sentences` is also `num_tokens`.

        # Shape: (batch_size, num_sentences, num_tokens, embedding_size)
        encoded = self._text_field_embedder(masked_sentences_tokens, num_wrapping_dims=1)

        # Does advanced indexing to get the embeddings of just the mask positions,
        # which is what we're trying to predict.
        batch_index = torch.arange(0, batch_size).unsqueeze(-1)
        sentence_index = torch.arange(0, num_sentences).unsqueeze(0)
        # Shape: (batch_size, num_sentences)
        mask_embeddings = encoded[batch_index, sentence_index, mask_positions]

        # Shape: (batch_size, num_sentences, vocab_size)
        logits = self._language_model_head(mask_embeddings)

        mask_positions_with_offsets = unmasked_sentence_tokens["bert-offsets"][
            batch_index, mask_positions
        ]
        original_token_indices = unmasked_sentence_tokens["bert"][
            batch_index, mask_positions_with_offsets
        ]

        # Shape: (batch_size, num_sentences)
        logits_original_tokens = logits[batch_index, sentence_index, original_token_indices]

        probs = F.softmax(logits_original_tokens, dim=-1)
        compliment = 1 - probs
        return {"unlikeliness": compliment / compliment.sum(dim=-1)}
