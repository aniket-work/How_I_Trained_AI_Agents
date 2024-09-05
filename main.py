#!/usr/bin/env python
import sys

from research_agents.crew import ResearchAgentsCrew


def run():
    print("Running Crew...")
    """
    Run the crew.
    """
    inputs = {
        'topic': "Do research on AAPL Stock"
    }
    print("calling ResearchAgentsCrew...")
    ResearchAgentsCrew().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'topic': "Do research on AAPL Stock"
    }
    try:
        ResearchAgentsCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        ResearchAgentsCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        'topic': "Do research on AAPL Stock"
    }
    try:
        ResearchAgentsCrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
