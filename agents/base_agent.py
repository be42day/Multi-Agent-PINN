from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def run(self, state: dict) -> dict:
        pass

    def run_with_state(self, state: dict) -> dict:
        new_data = self.run(state)
        return {**state.dict(), **new_data}
