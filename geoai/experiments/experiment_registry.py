from typing import Dict, List, Optional, Type
import numpy as np
from geoai.models.base import BaseModel, ModelCapabilities
from geoai.core.exceptions import ModelError
from geoai.models.baselines.rf_baseline import RFBaselineModel
from geoai.models.production.rf_enhanced import RFEnhancedModel
from geoai.models.baselines.extra_trees import ExtraTreesModel
from geoai.models.baselines.xgboost import XGBoostModel
from geoai.models.baselines.lightgbm import LightGBMModel
from geoai.models.baselines.catboost import CatBoostModel
from geoai.models.baselines.dl_wrapper import FCEFModel, FCSiamConcModel, FCSiamDiffModel, LightweightSiamCNNModel

class ModelImplementationUnavailableError(ModelError):
    """Exception raised when attempting to run training or inference on an unimplemented model."""
    def __init__(self, model_id: str) -> None:
        super().__init__(f"Model implementation not yet available: {model_id}")


class ResearchProvenance:
    """Strongly typed metadata tracking a model's scientific publication source."""
    def __init__(
        self,
        model_name: str,
        paper_title: str,
        authors: str,
        publication_year: int,
        venue: str,
        doi: str,
        repository: str,
        license: str,
        implementation_status: str,
        citation: str,
        current_validation_status: str
    ) -> None:
        self.model_name = model_name
        self.paper_title = paper_title
        self.authors = authors
        self.publication_year = publication_year
        self.venue = venue
        self.doi = doi
        self.repository = repository
        self.license = license
        self.implementation_status = implementation_status
        self.citation = citation
        self.current_validation_status = current_validation_status

    def to_dict(self) -> dict:
        return self.__dict__


class UnimplementedBaseModel(BaseModel):
    """Placeholder class for unimplemented models to enforce scientific integrity checks."""
    def __init__(self, model_id: str, provenance: ResearchProvenance) -> None:
        super().__init__()
        self.model_id = model_id
        self.provenance = provenance
        self.feature_names = []
        self.is_loaded = False

    def load(self, model_path) -> None:
        raise ModelImplementationUnavailableError(self.model_id)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise ModelImplementationUnavailableError(self.model_id)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise ModelImplementationUnavailableError(self.model_id)

    def get_feature_names(self) -> List[str]:
        return []

    def is_implemented(self) -> bool:
        return False

    def get_capabilities(self) -> ModelCapabilities:
        from geoai.models.base import ModelCapabilities
        channels = ["Red", "Green", "Blue"]
        if self.model_id in ("changeformer", "siamese"):
            channels = ["Red", "Green", "Blue", "SAR"]
        reqs = []
        if self.model_id in ("changeformer", "siamese"):
            reqs = ["torch", "torchvision"]
        elif self.model_id == "xgboost":
            reqs = ["xgboost"]
        elif self.model_id == "catboost":
            reqs = ["catboost"]
        elif self.model_id == "lightgbm":
            reqs = ["lightgbm"]
        return ModelCapabilities(
            model_type=self.provenance.model_name,
            expected_input_channels=channels,
            training_status="Not Yet Implemented",
            requirements=reqs
        )


# Concrete wrapper subclasses for scientific traceability


class SiameseNetworkModel(UnimplementedBaseModel):
    def __init__(self):
        prov = ResearchProvenance(
            model_name="Siamese Networks (Siam-FullyConv)",
            paper_title="Fully Convolutional Siamese Networks for Change Detection",
            authors="Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch",
            publication_year=2018,
            venue="ICIP",
            doi="10.1109/ICIP.2018.8451652",
            repository="https://github.com/rcdaudt/fully-convolutional-siamese-networks-for-change-detection",
            license="MIT",
            implementation_status="Not Yet Implemented",
            citation="""@inproceedings{daudt2018fully,
  title={Fully Convolutional Siamese Networks for Change Detection},
  author={Daudt, Rodrigo Caye and Le Saux, Bertrand and Boulch, Alexandre},
  booktitle={2018 25th IEEE International Conference on Image Processing (ICIP)},
  pages={4063--4067},
  year={2018},
  organization={IEEE}
}""",
            current_validation_status="Unverified"
        )
        super().__init__("siamese", prov)


class ChangeFormerModel(UnimplementedBaseModel):
    def __init__(self):
        prov = ResearchProvenance(
            model_name="ChangeFormer",
            paper_title="ChangeFormer: A Transformer-Based Change Detection Network for Earth Observation Images",
            authors="Wele Gedara Chaminda Bandara, Patel M. Patel",
            publication_year=2022,
            venue="IEEE IGARSS",
            doi="10.1109/IGARSS46834.2022.9883656",
            repository="https://github.com/wgcban/ChangeFormer",
            license="MIT",
            implementation_status="Not Yet Implemented",
            citation="""@inproceedings{bandara2022changeformer,
  title={Changeformer: A transformer-based change detection network for earth observation images},
  author={Bandara, Wele Gedara Chaminda and Patel, Vishwanath A},
  booktitle={IGARSS 2022-2022 IEEE International Geoscience and Remote Sensing Symposium},
  pages={124--127},
  year={2022},
  organization={IEEE}
}""",
            current_validation_status="Unverified"
        )
        super().__init__("changeformer", prov)


from geoai.models.transformers import TinyCD, BIT, Changer, ChangeFormer

# Main experiment models mappings
EXPERIMENT_MODEL_REGISTRY = {
    "rf_enhanced_v1": RFEnhancedModel,
    "rf_baseline_v1": RFBaselineModel,
    "extra_trees": ExtraTreesModel,
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel,
    "lightgbm": LightGBMModel,
    "siamese": SiameseNetworkModel,
    "changeformer": ChangeFormer,
    "fc_ef": FCEFModel,
    "fc_siam_conc": FCSiamConcModel,
    "fc_siam_diff": FCSiamDiffModel,
    "lightweight_siam_cnn": LightweightSiamCNNModel,
    "tinycd": TinyCD,
    "bit": BIT,
    "changer": Changer
}

def get_experiment_model(model_id: str) -> BaseModel:
    """Instantiate a registered model or unimplemented placeholder model."""
    cls = EXPERIMENT_MODEL_REGISTRY.get(model_id)
    if cls is None:
        raise ModelError(f"Model ID '{model_id}' is not registered in the Experiment Model Registry.")
    return cls()

def list_registered_models() -> List[str]:
    """List all registered experiment model identifiers."""
    return list(EXPERIMENT_MODEL_REGISTRY.keys())

