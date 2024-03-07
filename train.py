import random
from typing import Dict, List, Mapping, Tuple, Set

from entity_gym.env import *
from enn_trainer import TrainConfig, State, init_train_state, train#, load_checkpoint

from TensorRTS import TensorRTS

import hyperstate

#nickbot = load_checkpoint('./checkpoints/latest-step000000008192').state.agent
#nickbot = RogueNetAgent(model.state.agent)

@hyperstate.stateful_command(TrainConfig, State, init_train_state)
def main(state_manager: hyperstate.StateManager) -> None:
    train(state_manager=state_manager, env=TensorRTS) #, agent=nickbot

if __name__ == "__main__":  # This is to train
    main()