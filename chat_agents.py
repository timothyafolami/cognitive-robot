from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools
from langchain.tools import BaseTool
from typing import Optional,Type, Union
from langchain.pydantic_v1 import BaseModel, Field
import googlemaps
from langchain_community.utilities import SearchApiAPIWrapper
from dotenv import load_dotenv
load_dotenv()
import os
import requests


os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ['SERPAPI_API_KEY'] = os.getenv('SERP_API_KEY')
os.environ["SEARCHAPI_API_KEY"] = os.getenv("SEARCH_API_KEY")
google_api_key = os.getenv('GOOGLE_API_KEY')

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

class MapAssistantInput(BaseModel):
    initial_location: str = Field(description="The initial location of the user")
    destination: Optional[str] = Field(description="The destination of the user")
    mov_method: Optional[str] = Field(description="The method of movement")

class MapAssistantTool(BaseTool):
    name = "map_assistant"
    description = "This tool helps with getting information about the user movement"
    args_schema: Type[BaseModel] = MapAssistantInput

    def _run(self, initial_location: str, destination: Optional[str] = None, mov_method: Optional[str] = None) -> Union[dict, str]:
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
search = SearchApiAPIWrapper()
llm_math = load_tools(['llm-math'], llm=ChatOpenAI(temperature=0))

@tool
def google_map_ass(text):
  '''Used when there is a need to locate user and destination, then also get directions between user's and destination's location
  If a user mentions anything that needs to relate to his location you first of all use the get current location to get it's longitude and lattitude automatically.
  When there's an issue with getting the user location, just ask the user for it instead.
  '''
  # getting latitude and longitude
  try:
    latitude, longitude = get_current_location(google_api_key)
    if latitude is not None and longitude is not None:
      return (latitude, longitude)
  except:
    print("There's an error getting your location, kindly provide a text address instead")


  # use this when it comes to getting direction between two know places
  initial_location = '' # you get ths from the user
  destination = ''     # you get this based on users destination
  movement_method = ''  # you also ask the user for this.
  response = map_assistant_tool.run(initial_location, destination, movement_method)

  if response:
    return response

@tool
def google_search(text):
  '''Useful when there is a need to search the internet to get precise answers.
  You can also use this to get information about a location based on the longitude and latitude.
  Ensure to get the right information always.
  '''
  response = search.run(text)
  return response


@tool
def distance_calculation(text):
  '''This is used to calculate distance between places when needed.
  Make sure the text input is structured such that it can be understood by the tool.
  '''
  dist = llm_math.run(text)
  return dist