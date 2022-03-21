import os
import json
import argparse
import logging

from keras.optimizers import RMSprop
from keras.utils import plot_model
from pyfiction.simulators.games.machineofdeath_simulator import MachineOfDeathSimulator
from pyfiction.simulators.games.savingjohn_simulator import SavingJohnSimulator
from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode
from agents.ssaqn import SSAQNAgent
import agents.random as random_agent

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
An SSAQN agent that supports leave-one-out generalisation testing
"""


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent',
                        choices=["random", "ssaqn"],
                        help='The agent being executed',
                        type=str,
                        default="random")

    parser.add_argument('--simulator',
                        help='[0] for MachineOfDeathSimulator, [1] SavingJohnSimulator ',
                        choices=[0, 1],
                        type=int,
                        default=0)

    parser.add_argument('--log_folder',
                        help='a folder to store logs in, default is "logs"',
                        type=str,
                        default="logs")

    parser.add_argument('--ssaqn_hyperparameters',
                        help='a folder of the hyperparameters used for ssaqn agent.',
                        type=str,
                        default=None)

    args = parser.parse_args()
    return args


def default_ssaqn_hyperparameter():

    return {
        "lr": 0.0001, "embedding_dimensions": 16, "lstm_dimensions": 32, "dense_dimensions": 8,
        "epochs": 5,
        "episodes": 250, # Originally 8192...
        "batch_size": 256,
        "gamma": 0.95,
        "epsilon": 1,
        "epsilon_decay": 0.99,
        "prioritized_fraction": 0.25,
        "test_interval": 16,
        "test_steps": [1, 5, 1, 5, 1, 1]
    }


def execute_ssaqn(training_simulator, log_folder, log_prefix, hyperparameters_folder=None):
    hyperparameters = load_hyperparameters(hyperparameters_folder)

    agent = SSAQNAgent(train_simulators=training_simulator, log_folder=log_folder)

    for hyperparameter in hyperparameters:
        # Use default hyperparameters if the provided one does not have it...
        hyperparameter = {**default_ssaqn_hyperparameter(), **hyperparameter}

        # Load or learn the vocabulary (random sampling on this many games could be extremely slow)
        agent.initialize_tokens('vocabulary.txt')

        optimizer = RMSprop(hyperparameter["lr"])

        agent.create_model(embedding_dimensions=hyperparameter["embedding_dimensions"],
                           lstm_dimensions=hyperparameter["lstm_dimensions"],
                           dense_dimensions=hyperparameter["dense_dimensions"],
                           optimizer=optimizer)

        try:
            plot_model(agent.model, to_file='model.png', show_shapes=True)
        except ImportError as e:
            logger.warning("Couldn't print the model image: {}".format(e))

        for i in range(hyperparameter["epochs"]):
            logger.info('Epoch %s', i)
            agent.train_online(episodes=hyperparameter["episodes"],
                               batch_size=hyperparameter["batch_size"],
                               gamma=hyperparameter["gamma"],
                               epsilon=hyperparameter["epsilon"],
                               epsilon_decay=hyperparameter["epsilon_decay"],
                               prioritized_fraction=hyperparameter["prioritized_fraction"],
                               test_interval=hyperparameter["test_interval"],
                               test_steps=hyperparameter["test_steps"],
                               log_prefix=log_prefix)


def execute_random(simulator, file_name, log_folder):
    episodes = 510
    runs_per_episode = 1
    max_steps = 500
    n_trials = 5
    random_agent.run_experiment(n_trials, log_folder, simulator, file_name, episodes, runs_per_episode, max_steps)


def load_hyperparameters(hyperparameters_folder):
    def load_jsonfile(json_file):
        with open(json_file) as jf:
            return json.load(jf)

    if hyperparameters_folder is None:
        return [default_ssaqn_hyperparameter()]

    json_files = [
        f"{hyperparameters_folder}/{filename}"
        for filename in os.listdir(hyperparameters_folder)
        if filename.endswith(".json")
    ]

    return [load_jsonfile(json_file) for json_file in json_files]


def main(args):
    index_to_simulator = {
        0: MachineOfDeathSimulator(),
        1: SavingJohnSimulator()
    }

    simulator_name = {
        0: "machine-of-death",
        1: 'saving-john'
    }

    simulator_index = args.simulator
    log_folder = args.log_folder
    train_simulator = index_to_simulator[simulator_index]

    agent_type = {
        "random": lambda _: execute_random(
            simulator=train_simulator, file_name=f"random-{simulator_name[simulator_index]}", log_folder=log_folder),
        "ssaqn": lambda _: execute_ssaqn(
            training_simulator=train_simulator,
            log_folder=log_folder,
            log_prefix=simulator_name[simulator_index],
            hyperparameters_folder=args.ssaqn_hyperparameters
        )
    }

    if args.agent not in agent_type:
        raise ValueError(f"please select from the following agents: {list(agent_type.keys())}")

    logging.info(f"[{agent_type}-agent]: executing training on {simulator_name[simulator_index]}")
    agent_type[args.agent](args)


if __name__ == '__main__':
    main(arguments())
