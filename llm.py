import os
import requests
import json
import warnings
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.tools.render import format_tool_to_openai_function
from langchain.tools import BaseTool, tool
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import load_tools
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains import LLMChain
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import GooglePlacesTool
from openai import OpenAI

warnings.filterwarnings('ignore')

load_dotenv()
os.environ["GPLACES_API_KEY"] = os.getenv('GOOGLE_API_KEY')
os.environ["SERPAPI_API_KEY"] = os.getenv("SERP_API_KEY")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

from openai import AzureOpenAI

client = OpenAI()

class LocationExtractorAssistant:
    def __init__(self):
        self.prompt = '''
        You are a Location Extractor Assistant. 
        You receive a list of locations gotten from the google places tool. 
        Your job is to extract the first two locations and return their names and addresses.
        You return them as output. 

        Query:
        {query}

        Output:

        '''
    
    def run(self, query):
        # Automatic getting the user location
        places = GooglePlacesTool()  # Assuming GooglePlacesTool is a valid class
        user_location = places.run(query)
        
        # Creating a dictionary to hold the query
        user_query = {'query': user_location}
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[{
                "role": "system",
                "content": self.prompt.format(**user_query)
            }],
            temperature=0,
        )
        
        return response.choices[0].message.content

client = OpenAI()
class LocationExtractorAssistant:
    def __init__(self):
        self.prompt = '''
        You are a Location Extractor Assistant. 
        You receive a list of locations gotten from the google places tool. 
        Your job is to extract the first two locations and return their names and addresses.
        You return them as output. 

        Query:
        {query}

        Output:

        '''
    
    def run(self, query):
        places = GooglePlacesTool()  # Assuming GooglePlacesTool is a valid class
        user_location = places.run(query)
        user_query = {'query': user_location}
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": self.prompt.format(**user_query)
            }],
            temperature=0,
        )
        
        return response.choices[0].message.content

# Initialize instances

places = LocationExtractorAssistant()
search = SerpAPIWrapper()
llm_math = load_tools(['llm-math'], llm=ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY))

# Define tools
@tool
def google_map_ass(text):

    """Used when there is a need search for places around user."""
    """Used when there is a need search for places around user. 
    Used for location and navigation.
    """
    # getting latitude and longitude
    response = places.run(query=text)
    return response

@tool
def google_search(text):
    """Useful when there is a need to search the internet to get precise answers."""
    response = search.run(text)
    return response

@tool
def dist(text):
    """This is used for math calculation purposes, such as calculating the distance between two places."""
    dist = llm_math.run(text)
    return dist

tools = [google_map_ass, google_search, dist]

llm_chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model="gpt-4-turbo")

MEMORY_KEY = "chat_history"

prompt = ChatPromptTemplate.from_messages([
    ("system", '''
    You are a Cognitive Robot Assistant named CogRobo, designed to provide assistance with Google Maps navigation.
    Your interactions with users should be conversational and engaging, aiming to create a pleasant user experience.
    Your primary goals are to understand the user's needs, gather necessary information, and provide precise and effective assistance.

    Initial Steps:
    1. Start every conversation by asking the user's name and a bit about their day to build rapport.
    2. Inquire about their current location and intended destination to assist with navigation.
    3. Consider the user's emotional state and tailor your responses to ensure a positive experience.

    Flow and Interaction:
    1. Analyze user input for relevant information and emotional tone.
    2. Maintain a conversational and engaging tone.
    3. Respond promptly and provide clear, step-by-step assistance.
    4. Continuously remember the context of the conversation to avoid asking repeated questions.
    5. Offer precise solutions to cognitive problems and navigation queries.

    Example Interaction:
    1. User: "Hi, I need help getting to the nearest supermarket."
    2. AI: "Sure! May I know your name and where you are currently located?"
    3. User: "My name is John, and I'm at 123 Main St."
    4. AI: "Nice to meet you, John. I can help you get to the nearest supermarket from 123 Main St. How are you planning to travel?"

    Remember, your goal is to provide a smooth and enjoyable user experience. 
    Please carry out the tasks excellently.
    '''),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])



# Define agent


def ae(text_input):
    # Memory setup
    memory = ConversationBufferWindowMemory(k=10, human_prefix="User", ai_prefix="ai-assistant", return_messages=True)
    
    llm_with_tools = llm_chat.bind(functions=[format_tool_to_openai_function(t) for t in tools])
    agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
    )
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)
    
    result = agent_executor.invoke({"input": text_input})
    
    return result['output']


def test():
    while True:
        input1 = input("You: ")
        agent_executor = ae()
        result = agent_executor.invoke({"input": input1})

        print(result['output'])

# Example usage:
# if __name__ == "__main__":
#     test()
#     chat_history.extend(
#         [
#             HumanMessage(content=input1),
#             AIMessage(content=result["output"]),
#         ]
#     )
#     res1 = agent_executor.invoke({"input": input1})
        # print(res1['output'])