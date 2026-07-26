#!/usr/bin/env python3
"""A simple command-line question and answer loop."""


def main():
    """Prompt the user until a termination word is entered."""
    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        question = input("Q: ")

        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        print("A:")


if __name__ == "__main__":
    main()
