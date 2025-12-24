from abc import abstractmethod
from typing import Dict


class AiroaBasePolicy:
    @abstractmethod
    def step(self, obs: Dict) -> Dict:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
