"""Brain encoder registry used by configurable training entry points."""
from .encoder_layers import ResidualAdd, FlattenHead, BaseModel
from .eegproject import EEGProject
from .stae import STAE
from .temporal_encoders import EEGTemporalGateLayer, EEGTemporalReweightLayer
from .shallow_net import ShallowNet
from .deepnet import DeepNet
from .eegnet import EEGNet
from .tsconv import TSConv

BrainEncoder = EEGProject
EEGProjectLayer = EEGProject
Shallownet = ShallowNet
Deepnet = DeepNet
EEGnet = EEGNet
TSconv = TSConv

__all__ = ['ResidualAdd','FlattenHead','BaseModel','EEGProject','BrainEncoder',
           'EEGProjectLayer','STAE','EEGTemporalGateLayer','EEGTemporalReweightLayer',
           'ShallowNet','DeepNet','EEGNet','TSConv','Shallownet','Deepnet','EEGnet','TSconv']
