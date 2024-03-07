import random
from typing import Dict, List, Mapping, Tuple, Set

from entity_gym.env import *
from enn_trainer import TrainConfig, State, init_train_state, train, load_checkpoint, RogueNetAgent

from TensorRTS import TensorRTS

import hyperstate

model = load_checkpoint('./checkpoints')
nickbot = RogueNetAgent(model.state.agent)

@hyperstate.stateful_command(TrainConfig, State, init_train_state)
def main(state_manager: hyperstate.StateManager) -> None:
    train(state_manager=state_manager, env=TensorRTS, agent=nickbot)

if __name__ == "__main__":  # This is to train
    main()