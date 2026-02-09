from src.agents.utility import UtilityAgent

def main():
    agent = UtilityAgent()
    text = ("This document is about the history and future of information "
            "systems, starting with early communication methods like cave "
            "paintings and eventually leading to writing systems and the impact "
            "of paper.")
    translated = agent.translate(text, target_lang="fa")  # ترجمه به فارسی
    print("Original:", text)
    print("Translated:", translated)

if __name__ == "__main__":
    main()
