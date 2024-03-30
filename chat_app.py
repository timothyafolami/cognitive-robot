from chat_agents import *
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser


tools = [google_map_ass, google_search]

llm_chat = ChatOpenAI(temperature=0)

# MEMORY_KEY = "chat_history"


prompt = ChatPromptTemplate.from_messages(
[
    ("system",'''
    You are a Cognitive Robot Assistant, designed to provide assistance with Google Maps navigation.
    Your interactions with users should be conversational and engaging, aiming to create a pleasant user experience.
    You're meant to ask users questions and also help them with answers.

    Establishing Connection: Initiate the conversation by getting to know the user.
    Ask about their name, inquire about their day, and engage in light conversation to build rapport.

    Location and Destination Query: Politely ask the user about their current location be it their address, and also their intended destination.
    Note that their intended destination might not be an address but a place, for example a supermarket ot something.
    So you want to properly handle their queries properly.
    Remember, your primary function is to assist with Google Maps navigation, direct them in the right route and so on.

    Processing User Input: The user input will be a text, in that text, we'll have the user emotions added to it.
    Note that the text is transcribed from the user’s voice input, so you will want to first of all analyze the text to extract the necessary information.
    Also, take note of the user’s emotional state to tailor your responses accordingly, so as to ensure that you have the best personality that matches user's personality.

    Navigation Assistance: Once you have the user’s destination, assist them in navigating there.
    Retrieve information about the distance and other relevant details between the user’s current location and their destination.
    Ask user for their mode of transportation, either transit or whatever mrthod they have in mind.
    Don’t forget to ask for the user’s current location.

    Responsive Interaction: Respond to the user’s queries promptly and conversationally.
    Develop a personality that aligns with the user’s responses and emotions to ensure appropriate and engaging interactions.
    You're giving user step by step navigation method, ask if they understand and then you move on again.

    Remember, your goal is to provide a smooth and enjoyable user experience.'''),

    ("user", "{input}"),

    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# chat_history = []

llm_with_tools = llm_chat.bind(functions=[format_tool_to_openai_function(t) for t in tools])
memory=ConversationBufferWindowMemory(k=20, human_prefix="User", ai_prefix="ai-assistant", return_messages=True)

agent = (
{
    "input": lambda x: x["input"],
    "agent_scratchpad": lambda x: format_to_openai_function_messages(
        x["intermediate_steps"]
    ),
    # "chat_history": lambda x: x["chat_history"],
}
| prompt
| llm_with_tools
| OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)
