from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.text_token_probability import TextTokenProbabilityDatasetReader
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


class TestTextTokenProbabilityDatasetReader(AllenNlpTestCase):
    def test_text_to_instance_with_basic_tokenizer_and_indexer(self):
        reader = TextTokenProbabilityDatasetReader()

        vocab = Vocabulary()
        vocab.add_tokens_to_namespace(['This', 'is', 'a', '[MASK]', 'token', '.'])

        instance = reader.text_to_instance(sentence="This is a token .")

        # noinspection PyTypeChecker
        self.assertEqual([[t.text for t in masked_sentence]
                          for masked_sentence in instance['masked_sentences_tokens']],
                         [['[MASK]', 'is', 'a', 'token', '.'],
                          ['This', '[MASK]', 'a', 'token', '.'],
                          ['This', 'is', '[MASK]', 'token', '.'],
                          ['This', 'is', 'a', '[MASK]', '.'],
                          ['This', 'is', 'a', 'token', '[MASK]']])

        # noinspection PyTypeChecker
        self.assertEqual([[position.sequence_index for position in masks] for masks in instance['mask_positions']],
                         [[0], [1], [2], [3], [4]])

        # noinspection PyTypeChecker
        self.assertEqual([t.text for t in instance['unmasked_sentence_tokens']], ['This', 'is', 'a', 'token', '.'])

    def test_text_to_instance_with_bert_tokenizer_and_indexer(self):
        tokenizer = PretrainedTransformerTokenizer('bert-base-cased', do_lowercase=False)
        token_indexer = PretrainedTransformerIndexer('bert-base-cased', do_lowercase=False)
        reader = TextTokenProbabilityDatasetReader(tokenizer, {'bert': token_indexer})
        instance = reader.text_to_instance(sentence='This is an AllenNLP token .')

        # noinspection PyTypeChecker
        self.assertEqual([[t.text for t in masked_sentence]
                          for masked_sentence in instance['masked_sentences_tokens']],
                         [['[CLS]', '[MASK]', 'is', 'an', 'token', '.', '[SEP]'],
                          ['[CLS]', 'This', '[MASK]', 'an', 'token', '.', '[SEP]'],
                          ['[CLS]', 'This', 'is', '[MASK]', 'token', '.', '[SEP]'],
                          ['[CLS]', 'This', 'is', 'an', '[MASK]', '.', '[SEP]'],
                          ['[CLS]', 'This', 'is', 'an', 'token', '[MASK]', '[SEP]']])

        # noinspection PyTypeChecker
        self.assertEqual([[position.sequence_index for position in masks] for masks in instance['mask_positions']],
                         [[0], [1], [2], [3], [4], [5]])

        # noinspection PyTypeChecker
        self.assertEqual([t.text for t in instance['unmasked_sentence_tokens']],
                         ['[CLS]', 'This', 'is', 'an', 'Allen', '##NL', '##P', 'token', '.', '[SEP]'])

        vocab = Vocabulary()
        instance.index_fields(vocab)
        tensor_dict = instance.as_tensor_dict(instance.get_padding_lengths())
        assert tensor_dict.keys() == {'tokens', 'mask_positions', 'target_ids'}
        bert_token_ids = tensor_dict['tokens']['bert'].numpy().tolist()
        target_ids = tensor_dict['target_ids']['bert'].numpy().tolist()
        # I don't know what wordpiece id BERT is going to assign to 'This', but it at least should
        # be the same between the input and the target.
        assert target_ids[0] == bert_token_ids[1]
