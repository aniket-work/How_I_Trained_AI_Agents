import os,json
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from crewai_tools import tool
import yfinance as yf

load_dotenv()


@tool
def get_stock_info(ticker: str) -> str:
    """Fetch basic information about a stock"""
    stock = yf.Ticker(ticker)
    info = stock.info
    return f"Company Name: {info['longName']}\n" \
           f"Sector: {info['sector']}\n" \
           f"Industry: {info['industry']}\n" \
           f"Current Price: ${info['currentPrice']}\n" \
           f"52 Week High: ${info['fiftyTwoWeekHigh']}\n" \
           f"52 Week Low: ${info['fiftyTwoWeekLow']}"


@tool
def get_stock_financials(ticker: str) -> str:
    """Fetch key financial metrics for a stock"""
    stock = yf.Ticker(ticker)
    financials = stock.financials
    return f"Revenue: {financials.loc['Total Revenue'].iloc[0]}\n" \
           f"Net Income: {financials.loc['Net Income'].iloc[0]}\n" \
           f"Gross Profit: {financials.loc['Gross Profit'].iloc[0]}"

@CrewBase
class ResearchAgentsCrew():
    """ResearchAgents crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    settings_file = 'settings.json'

    def __init__(self) -> None:
        # Load settings
        with open(self.settings_file, 'r') as f:
            self.settings = json.load(f)

        # Initialize LLM based on settings
        if self.settings['llm'] == 'groq':
            self.llm = ChatGroq(
                temperature=0,
                groq_api_key=os.environ.get("GROQ_API_KEY"),
                model_name=self.settings['groq_model'],
            )
        elif self.settings['llm'] == 'ollama':
            self.llm = ChatOpenAI(
                model=self.settings['ollama_model'],
                base_url=self.settings['ollama_base_url'],
                api_key="ollama",
                temperature=0,
            )
        else:
            raise ValueError(f"Unsupported LLM: {self.settings['llm']}")

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[get_stock_info, get_stock_financials],  # Example of custom tool, loaded on the beginning of file
            llm=self.llm,
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            llm=self.llm,
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        print("Inside crew method")
        """Creates the ResearchAgents crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )