from unimodal_pair_encoders.hague import config
from unimodal_pair_encoders.hague.utils import get_feature_name_list


class PairProcessor:
    def __init__(self, property_dict, pairs_list):
        self.max_ratio_val = config.Features.max_ratio_val
        self.property_dict = property_dict
        self.pairs_list = pairs_list
        self.feature_name_list = get_feature_name_list()
        self.feature_funcs_dict = self._get_feature_dict()
        self.feature_to_att_dict = self._get_feature_to_att_dict()
        self.feature_vec = self._create_feature_vectors()

    def _get_feature_dict(self):
        return {feature: self._get_ratio for feature in self.feature_name_list}

    def _get_feature_to_att_dict(self):
        return {feature: feature.split('_ratio')[0] for feature in self.feature_name_list}

    def _get_feature_vec(self, pair):
        feature_vec = []
        for feature_name in self.feature_name_list:
            try:
                feature_vec.append(min(self.max_ratio_val,
                                       self.feature_funcs_dict[feature_name](feature_name, pair)))
            except:
                print(f'Error in feature {feature_name} for pair {pair}')
                feature_vec.append(0.0)
        return feature_vec

    def _create_feature_vectors(self):
        feature_vectors = []
        for pair in self.pairs_list:
            feature_vectors.append(self._get_feature_vec(pair))
        return feature_vectors

    def _get_ratio(self, feature_name, pair):
        att_name = self.feature_to_att_dict[feature_name]
        cand_att_val = self.property_dict[att_name]['cands'][pair[0]]
        index_att_val = self.property_dict[att_name]['index'][pair[1]]
        return round(cand_att_val / index_att_val, 3)

