
from poker_ai.agents.nash_agent import NashAgent
from poker_ai.agents.strategy_agent import StrategyAgent
from poker_ai.agents.strategy_io import load_strategy
from poker_ai.games.kuhn import KuhnPoker


# loads the selected opponent using either its name or menu number

# loads any of our saved poker strategies using its name or menu number
def make_opponent(selection):
    models = {
        "2": "cfr",
        "cfr": "cfr",
        "3": "cfr_plus",
        "cfr+": "cfr_plus",
        "cfr_plus": "cfr_plus",
        "4": "q_learning_random_decay_alpha",
        "q_learning_random": "q_learning_random_decay_alpha",
        "5": "q_learning_mixed_decay_alpha",
        "q_learning_mixed": "q_learning_mixed_decay_alpha",
        "6": "dqn_random",
        "dqn_random": "dqn_random",
        "7": "dqn_mixed",
        "dqn_mixed": "dqn_mixed",
        "8": "dqn_mixed_mse",
        "dqn_mixed_mse": "dqn_mixed_mse",
        "dqn": "dqn_mixed_mse"
    }

    selection = selection.strip().lower()

    # nash is built into the project so no saved file is needed
    if selection in ("1", "nash"):
        return NashAgent()

    if selection not in models:
        raise ValueError("Invalid opponent selection")

    # q learning and dqn use their saved greedy strategies for gameplay
    # the network weights and q tables are not needed for inference here
    filename = models[selection] + ".json"
    strategy = load_strategy(f"models/kuhn/{filename}")

    return StrategyAgent(strategy)



# accepts either the number or name of a legal action
def prompt_action(legal_actions):
    print("\nYour options:")
    for number, action in enumerate(legal_actions, start=1):
        print(f"{number}. {action.capitalize()}")

    while True:
        choice = input("Your choice: ").strip().lower()

        if choice in legal_actions:
            return choice

        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(legal_actions):
                return legal_actions[index]

        print("Invalid action. Enter a number or action name.")


# plays one hand against the selected agent
def play_hand(agent, human_player):
    game = KuhnPoker()

    print("\n--- New hand ---")
    print(f"Your position: Player {human_player}")
    print(f"Your card: {game.cards[human_player]}")

    while not game.done:
        # each player antes one chip and each bet or call adds one more
        pot = 2 + game.history.count("b") + game.history.count("k")
        print(f"\nPot: {pot} chips")

        if game.current_player == human_player:
            action = prompt_action(game.legal_actions())
        else:
            action = agent.choose_action(
                game.get_information_set(),
                game.legal_actions()
            )
            print(f"AI chooses: {action}")

        game.step(action)

    # payoffs already include the chips committed to the pot
    payoff = game.get_payoff()[human_player]

    print(f"\nAI card: {game.cards[1 - human_player]}")
    print(f"Hand payoff: {payoff:+d} chips")

    return payoff


def main():
    print("\nChoose an opponent:")
    print("1. Nash Equilibrium")
    print("2. CFR")
    print("3. CFR+")
    print("4. Q-learning (Random)")
    print("5. Q-learning (Mixed)")
    print("6. DQN (Random, Huber)")
    print("7. DQN (Mixed, Huber)")
    print("8. DQN (Mixed, MSE)")


    try:
        agent = make_opponent(input("Selection: "))
    except (ValueError, FileNotFoundError) as error:
        print(error)
        return

    total_payoff = 0
    hand_count = 0
    wins = 0

    while True:
        # alternate player positions to reduce seat imbalance
        human_player = hand_count % 2
        payoff = play_hand(agent, human_player)

        total_payoff += payoff
        hand_count += 1

        if payoff > 0:
            wins += 1

        print(f"\nHands: {hand_count} | Wins: {wins} | Losses: {hand_count - wins}")
        print(f"Total profit: {total_payoff:+d} chips")

        if input("Play again? (y/n): ").strip().lower() not in ("y", "yes"):
            break


if __name__ == "__main__":
    main()
