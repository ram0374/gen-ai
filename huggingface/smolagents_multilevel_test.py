
import os
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, tool
os.environ["HF_TOKEN"]=""
# agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct"))

# agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")

# Tool to suggest a menu based on the occasion
@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion (str): The type of occasion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero party.
                        - "custom": Custom menu.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."

# Alfred, the butler, preparing the menu for the party
#agent = CodeAgent(tools=[suggest_menu,], model=HfApiModel())

# Preparing the menu for the party

#agent.run("Prepare a menu for the party.")

import numpy as np
import time
import datetime

agent = CodeAgent(tools=[], model=HfApiModel(), additional_authorized_imports=['datetime'])

# agent.run(
#     """
#     Alfred needs to prepare for the party. Here are the tasks:
#     1. Prepare the drinks - 30 minutes
#     2. Decorate the mansion - 60 minutes
#     3. Set up the menu - 45 minutes
#     4. Prepare the music and playlist - 45 minutes

#     If we start right now, at what time will the party be ready?
#     """
# )
agent.push_to_hub('RamPasupula/PartyAgent')