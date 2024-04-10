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
from typing import Optional, Type, Union
from langchain.pydantic_v1 import BaseModel, Field
import googlemaps
from datetime import datetime
from langchain.agents import AgentType, Tool, initialize_agent
from langchain_community.utilities import SearchApiAPIWrapper
import warnings
from flask import Flask, request, jsonify

warnings.filterwarnings('ignore')

# google_api_key = os.environ.get('GOOGLE_API_KEY')
# print(google_api_key)
# os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')
# os.environ["SEARCH_API_KEY"] = os.environ.get("SEARCH_API_KEY")
google_api_key = 'AIzaSyBnSXBGH52zDDvz1a-0pxbLT-kBTG_3U3M'
OPENAI_API_KEY = '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'
SEARCH_API_KEY = '1N4PnjsUL9nBddCWag61KJMq'


def get_current_location(api_key):
    url = f'https://www.googleapis.com/geolocation/v1/geolocate?key={api_key}'
    response = requests.post(url)
    if response.status_code == 200:
        data = response.json()
        latitude = data['location']['lat']
        longitude = data['location']['lng']
        return latitude, longitude
    else:
        print("Failed to retrieve location.")
        return None, None


# latitude, longitude = get_current_location(google_api_key)

class MapAssistantInput(BaseModel):
    initial_location: str = Field(description="The initial location of the user")
    destination: Optional[str] = Field(description="The destination of the user")
    mov_method: Optional[str] = Field(description="The method of movement")


class MapAssistantTool(BaseTool):
    name = "map_assistant"
    description = "This tool helps with getting information about the user movement"
    args_schema: Type[BaseModel] = MapAssistantInput

    def _run(self, initial_location: str, destination: Optional[str] = None, mov_method: Optional[str] = None) -> Union[
        dict, str]:
        gmaps = googlemaps.Client(key=google_api_key)

        # Geocoding an address
        geocode_result = gmaps.geocode(initial_location)

        # Request directions via public transit
        now = datetime.now()
        directions_result = gmaps.directions(initial_location,
                                             destination,
                                             mode=mov_method,
                                             departure_time=now)
        if geocode_result:
            return geocode_result
        if directions_result:
            return directions_result


map_assistant_tool = MapAssistantTool()
search = SearchApiAPIWrapper(searchapi_api_key=SEARCH_API_KEY)
llm_math = load_tools(['llm-math'], llm=ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY))


@tool
def google_map_ass(text):
    """Used when there is a need to locate user and destination, then also get directions between user's and
    destination's location If a user mentions anything that needs to relate to his location you first of all use the
    get current location to get it's longitude and lattitude automatically. When there's an issue with getting the
    user location, just ask the user for it instead.
    """
    # getting latitude and longitude
    try:
        latitude, longitude = get_current_location(google_api_key)
        if latitude is not None and longitude is not None:
            return (latitude, longitude)
    except:
        print("There's an error getting your location, kindly provide a text address instead")

    # use this when it comes to getting direction between two know places
    initial_location = ''  # you get ths from the user
    destination = ''  # you get this based on users destination
    movement_method = ''  # you also ask the user for this.
    response = map_assistant_tool.run(initial_location, destination, movement_method)

    if response:
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
def distance_calculation(text):
    """This is used to calculate distance between places when needed.
  Make sure the text input is structured such that it can be understood by the tool.
  """
    dist = llm_math.run(text)
    return dist


tools = [google_map_ass, google_search]

llm_chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

MEMORY_KEY = "chat_history"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", '''
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

chat_history = []

llm_with_tools = llm_chat.bind(functions=[format_tool_to_openai_function(t) for t in tools])
memory = ConversationBufferWindowMemory(k=10, human_prefix="User", ai_prefix="ai-assistant", return_messages=True)

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


def ae():
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)
    return agent_executor


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


# app = Flask(__name__)


# @app.route('/process_llm_text', methods=['POST'])
# def process_text():
#     data = request.get_json()
#     input_text = data['input_text']
#     agent_executor = ae()
#     result = agent_executor.invoke({"input": input_text})
#     return jsonify({'output': result['output']})


# if __name__ == '__main__':
#     app.run(debug=True)