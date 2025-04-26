# from beartype.vale import Is  # <---- validator factory
# from typing import Annotated  # <---------------- if Python ≥ 3.9.0
from river.base import Dataset, Metric, Estimator
from expectation.modules.hypothesistesting import EProcess

Reals = Union[int, float]
__all__ = ["Dataset", "Metric", "Estimator", "Reals", "EProcess"]
