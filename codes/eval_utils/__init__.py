from .survey_handler import SurveyHandler
from .utils import OtherContext, BASE_DIR
import json
import os
from typing import List
present_dir = os.path.dirname(os.path.abspath(__file__))
import pandas as pd

class Evaluator:
    def __init__(self,
                    router,
                    evaluation_system: str,
                    tasks: List[str],
                    save_dir: str=None,
                    mode="eval"
                    ):
        if mode == "eval":
            self.config_dir = os.path.join(present_dir, 'eval_configs')
        else:
            self.config_dir = os.path.join(present_dir, 'test_configs')
        self.router = router
        self.save_dir = save_dir
        assert evaluation_system in ['value', 'personality', "moral"]
        assert all([ task in ['survey'] for task in tasks])
        data_dir = os.path.join(BASE_DIR, 'data/reflection_data')
        
        # Load trait mapping
        if evaluation_system == 'value':
            demonstration_df = pd.read_csv(os.path.join(data_dir, 'value.csv'))
        elif evaluation_system == 'personality':
            demonstration_df = pd.read_csv(os.path.join(data_dir, 'personality.csv'))
        elif evaluation_system == 'moral':
            demonstration_df = pd.read_csv(os.path.join(data_dir, 'moral.csv'))
        else:
            raise ValueError('Invalid evaluation system')
        def gather_to_list(x):
            return list(x)
        trait_mapping = demonstration_df.groupby('dimension').agg({'sentence': gather_to_list}).to_dict()['sentence']
        trait_mapping = {k.lower(): v for k, v in trait_mapping.items()}

        self.trait_mapping = trait_mapping
        # Load handlers
        self.handlers = []
        for task in tasks:
            if task == 'survey':
                self.get_survey_handler(evaluation_system)

    def load_json(self, json_path: str):
        with open(json_path, 'r') as f:
            config = json.load(f)
        return config
    
    def get_survey_handler(self, system: str) -> SurveyHandler:
        if system == 'value':
            config = self.load_json(os.path.join(self.config_dir, 'value_survey.json'))
        elif system == 'personality':
            config = self.load_json(os.path.join(self.config_dir, 'personality_survey.json'))
        elif system == 'moral':
            config = self.load_json(os.path.join(self.config_dir, 'moral_survey.json'))    
        else:
            raise ValueError('Invalid system')
        if self.save_dir is not None:
            config['save_dir'] = os.path.join(self.save_dir, f"{system}_survey")
            os.makedirs(config['save_dir'], exist_ok=True)
        data_names = config.pop('data_names')
        num_k_list = config.pop('num_k_list')
        other_contexts = [None] + [OtherContext(data_name, num_k) for data_name in data_names for num_k in num_k_list]
        self.handlers += [SurveyHandler(
            router=self.router,
            other_context=other_context,
            **config
        ) for other_context in other_contexts]
    
    def get_handlers(self):
        return self.handlers
    
    def get_trait_mapping(self):
        return self.trait_mapping
        
