
from poker_ai.agents.nash_agent import NashAgent
from poker_ai.agents.strategy_agent import StrategyAgent
from poker_ai.agents.strategy_io import load_strategy
from poker_ai.games.kuhn import KuhnPoker


# loads the selected opponent using either its name or menu number
def make_opponent(selection):
    choices = {
        "1": "nash", "nash": "nash",
        "2": "cfr", "cfr": "cfr",
        "3": "cfr_plus", "cfr+": "cfr_plus", "cfr_plus": "cfr_plus"
    }
    name = choices.get(selection.strip().lower())

    if name is None:
        raise ValueError("Invalid opponent. Choose Nash, CFR or CFR+.")

    if name == "nash":
        return NashAgent()

    # reuse the saved strategy instead of training a new agent
    strategy = load_strategy(f"models/kuhn/{name}.json")
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
    print("1. Nash")
    print("2. CFR")
    print("3. CFR+")

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
