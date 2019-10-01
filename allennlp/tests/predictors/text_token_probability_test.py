from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


class TestTextTokenProbabilityPredictor(AllenNlpTestCase):
    def test_predictions_to_labeled_instances(self):
        input_ = {'sentence': "A busy barber is quite harried"}

        archive = load_archive(self.FIXTURES_ROOT / 'text_token_probability' / 'serialization' / 'model.tar.gz')
        predictor = Predictor.from_archive(archive, 'text_token_probability')

        instance = predictor._json_to_instance(input_)
        output = predictor._model.forward_on_instance(instance)
        self.assertEqual(output['probabilities'].shape, (6,))
        # new_instances = predictor.predictions_to_labeled_instances(instance, output)
        # assert len(new_instances) == 1
        # assert 'target_ids' in new_instances[0]
        # assert len(new_instances[0]['target_ids'].tokens) == 2  # should have added two words
