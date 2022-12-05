from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from urllib.parse import uses_params

import pandas as pd

# from ..model import Model


@dataclass
class Evaluation:
    models: Dict[str, any] = field(default_factory=dict)

    def run(
        self, test: pd.DataFrame, metrics: Dict, save_dir=None
    ) -> List[Tuple[str, pd.DataFrame]]:
        df = pd.DataFrame(columns=list(metrics.keys()))
        for name, model in self.models.items():
            # Evaluate model
            row = pd.Series(model.predict(test.copy(), metrics=metrics))
            row.name = name
            df = df.append(row)
        if save_dir:
            df.to_csv(save_dir + "/" + "results" + ".csv")
        return df

    def register_model(self, name, model):
        """
        Register Model for Evaluation.
        Model must inherit from base class Model and must be already trained.
        """
        self.models[name] = model
