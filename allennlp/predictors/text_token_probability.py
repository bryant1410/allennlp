from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('text_token_probability')
class TextTokenProbabilityPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({'sentence': sentence})

    # @overrides
    # def predictions_to_labeled_instances(self, instance: Instance,
    #                                      outputs: Dict[str, numpy.ndarray]) -> Iterable[Instance]:
    #     new_instance = deepcopy(instance)
    #     token_field: TextField = instance['tokens']
    #     mask_targets = [Token(target_top_k[0]) for target_top_k in outputs['words']]
    #     new_instance.add_field('target_ids',
    #                            TextField(mask_targets, token_field._token_indexers),
    #                            vocab=self._model.vocab)
    #     return [new_instance]

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """Expects JSON like ``{"sentence": "..."}``."""
        return self._dataset_reader.text_to_instance(sentence=json_dict['sentence'])
