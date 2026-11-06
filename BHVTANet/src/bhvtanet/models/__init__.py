"""Paper-named models; implementations are imported on demand."""
from importlib import import_module

_MODEL_MODULES = {'BHVTANet':'bhvtanet', 'MCTF':'mctf', 'MDTP':'mdtp',
                  'DACAM':'dacam', 'DSWR':'dswr', 'EEGProject':'eegproject',
                  'STAE':'stae', 'ShallowNet':'shallow_net', 'DeepNet':'deepnet',
                  'EEGNet':'eegnet', 'TSConv':'tsconv'}
__all__ = list(_MODEL_MODULES)

def __getattr__(name):
    if name not in _MODEL_MODULES:
        raise AttributeError(name)
    return getattr(import_module(f'{__name__}.{_MODEL_MODULES[name]}'), name)
