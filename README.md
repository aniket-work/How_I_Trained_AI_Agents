
# How I Trained AI Agents


## Introduction

Full Article : [https://medium.com/@learn-simplified/how-i-trained-ai-agents-why-you-should-too-5beda74dc57c

AI agents has become a crucial step in maximizing potential and effectiveness. As businesses and organizations increasingly rely on AI technologies, the importance of well-trained AI agents cannot be overstated.

## Why Train AI Agents?

Training AI agents is essential for several compelling reasons:

 - Improved decision-making: Through training, AI agents learn to make more informed and accurate decisions based on past experiences and learned patterns. This leads to better outcomes in complex scenarios where nuanced understanding is required.
 - Enhanced adaptability: Trained AI agents can adapt to changing environments and novel situations more effectively. This flexibility is vital in dynamic real-world settings where conditions may vary unpredictably.
 - Increased efficiency: Training enables AI agents to optimize their processes and workflows, resulting in faster task completion and better resource utilization. This efficiency translates to significant time and cost savings for businesses.
 - Continuous improvement: Training is an ongoing process that allows AI agents to evolve and improve over time. This ensures that the agents remain relevant and effective as new data and scenarios emerge.

## Untrained and Trained AI Agents In A Real-World 

Untrained AI Agent Response:

```text
Customer: “I received a damaged product. What should I do?”
Untrained Agent: “I’m sorry to hear that. Please contact our customer support team for assistance.”
```

This response is generic, not tailored to the specific situation, and fails to provide actionable information to resolve the customer’s issue.

Trained AI Agent Response:

```text
Customer: “I received a damaged product. What should I do?”
Trained Agent: “I apologize for the inconvenience. Here are the steps to resolve this issue:


Take clear photos of the damaged product.
Go to your order history and select the affected item.
Click on ‘Report an Issue’ and choose ‘Damaged Product’.
Upload the photos and provide a brief description.
Our team will review your claim within 24 hours and arrange a replacement or refund.
Is there anything else I can help you with regarding this issue?”
```

The trained agent’s response demonstrates:

Better understanding of the context
More detailed and helpful information
A proactive approach to problem-solving
Improved customer experience

This example highlights how training can significantly enhance the quality and effectiveness of AI agent interactions, leading to superior outcomes for both users and businesses


## Architecture
![Design Diagram](design_docs/design.png)


# Tutorial: Let's Build AI Agents That Code for You

## Prerequisites
- Python installed on your system.
- A basic understanding of virtual environments and command-line tools.

## Steps

1. **Virtual Environment Setup:**
   - Create a dedicated virtual environment for our project:
   
     ```bash
     python -m venv How_I_Trained_AI_Agents
     ```
   - Activate the environment:
   
     - Windows:
       ```bash
       How_I_Trained_AI_Agents\Scripts\activate
       ```
     - Unix/macOS:
       ```bash
       source How_I_Trained_AI_Agents/bin/activate
       ```


   
# AI Agent Training Installation and Setup Guide

**Install Project Dependencies:**

Follow these steps to set up and run the ResearchAgents project:

1. Navigate to your project directory:
   ```
   cd path/to/your/project
   ```
   This ensures you're in the correct location for the subsequent steps.

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
   This command installs all the necessary Python packages listed in the requirements.txt file.

3. Create a new crew using crewai:
   ```
   crewai create crew research_agents
   ```
   This command sets up the initial structure for your ResearchAgents crew.

4. Copy the main files from the git repository:
   - Copy `main.py` and `crew.py` from the git repo into the autogenerated `research_agents/src/research_agents` directory.
   These files contain the core functionality of the ResearchAgents project.

5. Set up the environment variables:
   - Copy the content of `.env.example` into a new `.env` file under the autogenerated `research_agents/src/research_agents` directory.
   - Add your Groq API key to the `.env` file.
   This step ensures that your API key is securely stored and accessible to the application.

6. Add the settings file:
   - Copy `settings.json` into the autogenerated `research_agents/src/research_agents` directory.
   This file contains important configuration settings for the project.

7. Navigate to the research_agents directory:
   ```
   cd research_agents
   ```
   This positions you in the correct directory for the following Poetry commands.

8. Install Poetry:
   ```
   pip install poetry
   ```
   Poetry is used for dependency management and packaging in Python.

9. Add the langchain_groq package:
   ```
   poetry add langchain_groq
   ```
   This adds the langchain_groq package to your project dependencies.

10. Update and install dependencies:
    ```
    poetry lock
    poetry install
    ```
    These commands update the lock file with the new dependency and install all project dependencies in a virtual environment.

By following these steps, you'll have a fully set up and configured ResearchAgents project ready to run. This process ensures that all necessary components are in place, including the core files, environment variables, settings, and required packages.   
     

**Install Ollama**
    
    Ollama is a powerful tool for running large language models locally on your machine. Let's walk through the installation process step-by-step.
    
    Step 1: Download Ollama
     - Visit the official Ollama website at https://ollama.com/ and click the "Download" button. The website will automatically detect your operating system and offer the appropriate installer
    
    Step 2: Install Ollama
      - For Windows and Mac users: Double-click the downloaded installer file (.exe for Windows, .dmg for Mac) and follow the on-screen instructions
      - For Linux users: Open a terminal and run the following command:

## Run - AI Agent Training

   ```bash 
   # Run AI Agent
   (How_I_Trained_AI_Agents) C:\Users\worka\PycharmProjects\How_I_Trained_AI_Agents\research_agents>crewai run
   
   # Train AI Agent 
   (How_I_Trained_AI_Agents) C:\Users\worka\PycharmProjects\How_I_Trained_AI_Agents\research_agents>crewai train -n 4   
   ```






