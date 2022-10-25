import random
import unittest

from tests.models.vit_mae.test_modeling_vit_mae import *
from transformers import ViTMAEAdapterModel
from transformers.testing_utils import require_torch

from .methods import (
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
)
from .test_adapter import VisionAdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .test_adapter_composition import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin
from .test_common import AdapterModelTesterMixin


@require_torch
class ViTMAEAdapterModelTest(AdapterModelTesterMixin, ViTMAEModelTest):
    all_model_classes = (
        ViTMAEAdapterModel,
    )
    fx_compatible = False


class ViTMAEAdapterTestBase(VisionAdapterTestBase):
    config_class = ViTMAEConfig
    config = make_config(
        ViTMAEConfig,
        image_size=224,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )
    feature_extractor_name = 'facebook/vit-mae-base'


@require_torch
class ViTMAEAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    ViTMAEAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class ViTMAEClassConversionTest(
    ModelClassConversionTestMixin,
    ViTMAEAdapterTestBase,
    unittest.TestCase,
):
    pass
