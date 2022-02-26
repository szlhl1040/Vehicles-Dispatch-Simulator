from abc import ABCMeta, abstractmethod


class DispatchModuleInterface(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self):
        raise NotImplementedError


class RandomDispatch(DispatchModuleInterface):
    def __init__(self):
        pass

    def __call__(self):
        pass
