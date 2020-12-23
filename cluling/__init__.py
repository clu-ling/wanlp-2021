try:
  from .info import info
  #from .utils import Dataset, ExperimentConfig
  #from .arabic.models import AraBERT
  __version__ = info.version
except:
  pass
