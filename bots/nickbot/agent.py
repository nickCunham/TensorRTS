import sys
import os
tensor_path = os.path.abspath(os.path.join(os.path.basename(__file__), os.pardir, os.pardir, os.pardir))
sys.path.append(tensor_path)

import random 
from typing import Dict, List, Mapping, Tuple, Set
from entity_gym.env import Observation
from entity_gym.runner import CliRunner
from entity_gym.env import *
from TensorRTS import Agent
from enn_trainer import load_checkpoint, RogueNetAgent

class nickbot(Agent):
    def __init__(self, init_observation : Observation, action_space : Dict[ActionName, ActionSpace]) -> None: 
        super().__init__(init_observation, action_space)
        model = load_checkpoint('../../checkpoints')
        self.current = RogueNetAgent(model.state.agent)

    def take_turn(self, current_game_state : Observation) -> Mapping[ActionName, Action]:
        mapping = self.current.act(current_game_state)
        return mapping
    
    def on_game_start(self) -> None:
        return super().on_game_start()
    
    def on_game_over(self, did_i_win : bool, did_i_tie : bool) -> None:
        return super().on_game_over(did_i_win, did_i_tie)
    
def agent_hook(init_observation : Observation, action_space : Dict[ActionName, ActionSpace]) -> Agent: 
    """Creates an agent of this type

    Returns:
        Agent: nickbot!
    """
    return nickbot(init_observation, action_space)

def student_name_hook() -> str: 
    """Provide the name of the student as a string

    Returns:
        str: Name of student
    """
    return 'Nick Cunningham'