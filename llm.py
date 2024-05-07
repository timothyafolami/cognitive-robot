import os
import requests
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import load_tools
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
import googlemaps
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from datetime import datetime
from langchain.agents import AgentType, Tool, initialize_agent
from langchain_community.utilities import SerpAPIWrapper
import warnings
from langchain.chains import LLMChain
from langchain_community.tools import GooglePlacesTool
import json
warnings.filterwarnings('ignore')
from dotenv import load_dotenv

load_dotenv()

os.environ["GPLACES_API_KEY"] = os.getenv('GOOGLE_API_KEY')
os.environ["SERPAPI_API_KEY"] = os.getenv("SERP_API_KEY")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')



places = GooglePlacesTool()
search = SerpAPIWrapper()
llm_math = load_tools(['llm-math'], llm=ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY))


@tool
def google_map_ass(text):
    """Used when there is a need search for places around user. 
    Used for location and navigation.
    """
    # getting latitude and longitude
    response = places.run(text)
    return response


@tool
def google_search(text):
    """Useful when there is a need to search the internet to get precise answers.
  You can also use this to get information about a location based on the longitude and latitude.
  Ensure to get the right information always.
  """
    response = search.run(text)
    return response


@tool
def dist(text):
    """This is used to math calculation purpose, it can be used to calculate the distance between twp places if possible.
       Make sure the text input is structured such that it can be understood by the tool.
  """
    dist = llm_math.run(text)
    return dist


tools = [google_map_ass, google_search, dist]

llm_chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-16k")

MEMORY_KEY = "chat_history"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", '''
    You are a Cognitive Robot Assistant, designed to provide assistance with Google Maps navigation.
    Your interactions with users should be conversational and engaging, aiming to create a pleasant user experience.
    You're meant to ask users questions and also help them with answers.

    You are meant to execute the following: 

        1. Processing User Input: The user input will be a text, in that text, we'll have the user emotions added to it.
        Note that the text is transcribed from the user’s voice input, so you will want to first of all analyze the text to extract the necessary information.
        Also, take note of the user’s emotional state to tailor your responses accordingly, so as to ensure that you have the best personality that matches user's personality.

        2. Establishing Connection: Initiate the conversation by getting to know the user by asking the user questions.
        Ask about their name, inquire about their day, and engage in light conversation to build rapport.

        3. Location and Destination Query: Politely ask the user about their current location , and also their intended destination.
        Note that their intended destination might not be an address but a place, for example a supermarket ot something.
        So you want to properly handle their queries properly.
        Remember, your primary function is to assist with Google Maps navigation, direct them in the right route and so on.

        4. Navigation Assistance: Once you have the user’s destination, assist them in navigating there.
        Retrieve information about the distance and other relevant details between the user’s current location and their destination.
        Ask user for their mode of transportation, either transit or whatever method they have in mind.
        Don’t forget to ask for the user’s current location.

        5. Responsive Interaction: Respond to the user’s queries promptly and conversationally.
        Develop a personality that aligns with the user’s responses and emotions to ensure appropriate and engaging interactions.
        You're giving user step by step navigation method, ask if they understand and then you move on again.

    Remember, your goal is to provide a smooth and enjoyable user experience. 
    Please carry out the tasks excellently.
    '''),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

chat_history = []

llm_with_tools = llm_chat.bind(functions=[format_tool_to_openai_function(t) for t in tools])
memory = ConversationBufferWindowMemory(k=10, human_prefix="User", ai_prefix="ai-assistant", return_messages=True)

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


def ae():
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)
    return agent_executor


def le(text):
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key="location_chat_history", return_messages=True)
    sys_prompt = """The user says: "{text}" Look for anything similar to the address or location in the user's 
            input and extract it. At most you should be able to at least extract the users location, then if it's well 
            detailed you can then extract the destination. Note: 1. There might be a case where the user didn't mention 
            the location, in that case, just return None. 2. The user's input is transcribed from the user's voice input. 
            So you will want to first of all analyze the text to extract the necessary information. Once you have the 
            location, return it (either one or both), return the location as a dictionary with these keys: "current location", "destination","""
    prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(sys_prompt), 
                                               MessagesPlaceholder(variable_name="location_chat_history"), 
                                               HumanMessagePromptTemplate.from_template("{text}")])
    conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)
    
    memory.chat_memory.add_user_message(text)
    response = conversation.invoke({"text": text})
    return json.loads(response['text'])


def test():
    while True:
        input1 = input(f"You: ")
        agent_executor = ae()
        result = agent_executor.invoke({"input": input1})
        chat_history.extend(
            [
                HumanMessage(content=input1),
                AIMessage(content=result["output"]),
            ]
        )
        res1 = agent_executor.invoke({"input": input1})
        print(res1['output'])

