let map;
function openDrawer() {
  const drawer = document.getElementById("drawer-example");
  drawer.setAttribute("data-drawer-show", "drawer-example");
  drawer.setAttribute("aria-expanded", "true");
  drawer.style.transform = "translateX(0)";
}
// Close drawer function
function closeDrawer() {
  const drawer = document.getElementById("drawer-example");
  drawer.removeAttribute("data-drawer-show");
  drawer.setAttribute("aria-expanded", "false");
  drawer.style.transform = "translateX(-100%)";
}
function openDropdown() {
  let dropDown = document.getElementById("dropDownDetails")
  if (dropDown.style.height = "0px"){
    dropDown.style.height = "100%"
  }else {
    dropDown.style.height = "0px"
  }
  
}
// Listen for clicks outside the drawer to close it
window.addEventListener("click", function (event) {
  const drawer = document.getElementById("drawer-example");
  if (event.target !== drawer && !drawer.contains(event.target)) {
    closeDrawer();
  }
});

// Prevent closing the drawer when clicking inside it
document
  .getElementById("drawer-example")
  .addEventListener("click", function (event) {
    event.stopPropagation();
  });
async function initMap() {
  const { Map, places } = await google.maps.importLibrary("maps");

  map = new Map(document.getElementById("map"), {
    center: { lat: -34.397, lng: 150.644 },
    zoom: 3,
  });

  const directionsService = new google.maps.DirectionsService();
  const directionsRenderer = new google.maps.DirectionsRenderer();

  directionsRenderer.setMap(map);

  const originInput = document.getElementById("origin-input");
  const destinationInput = document.getElementById("destination-input");

  const autocompleteOrigin = new google.maps.places.Autocomplete(
    originInput
  );
  const autocompleteDestination = new google.maps.places.Autocomplete(
    destinationInput
  );

  autocompleteOrigin.bindTo("bounds", map);
  autocompleteDestination.bindTo("bounds", map);

  // Add a listener for the directions button
  document
    .getElementById("get-directions")
    .addEventListener("click", function () {
      directionsService.route(
        {
          origin: originInput.value,
          destination: destinationInput.value,
          travelMode: "DRIVING",
        },
        function (response, status) {
          if (status === "OK") {
            directionsRenderer.setDirections(response);
          } else {
            window.alert("Directions request failed due to " + status);
          }
        }
      );
    });

  infoWindow = new google.maps.InfoWindow();

  const locationButton = document.createElement("button");

  locationButton.textContent = "Pan to Current Location";
  locationButton.classList.add("custom-map-control-button");
  map.controls[google.maps.ControlPosition.TOP_CENTER].push(
    locationButton
  );
  locationButton.addEventListener("click", () => {
    // Try HTML5 geolocation.
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const pos = {
            lat: position.coords.latitude,
            lng: position.coords.longitude,
          };

          infoWindow.setPosition(pos);
          infoWindow.setContent("Location found.");
          infoWindow.open(map);
          map.setCenter(pos);
        },
        () => {
          handleLocationError(true, infoWindow, map.getCenter());
        }
      );
    } else {
      // Browser doesn't support Geolocation
      handleLocationError(false, infoWindow, map.getCenter());
    }
  });

  function handleLocationError(browserHasGeolocation, infoWindow, pos) {
    infoWindow.setPosition(pos);
    infoWindow.setContent(
      browserHasGeolocation
        ? "Error: The Geolocation service failed."
        : "Error: Your browser doesn't support geolocation."
    );
    infoWindow.open(map);
  }

  // Create the search box and link it to the UI element.
  const input = document.getElementById("searchInput");
  const searchBox = new google.maps.places.SearchBox(input);

  map.controls[google.maps.ControlPosition.TOP_LEFT].push(input);
  // map.controls[google.maps.ControlPosition.TOP_LEFT].push(originInput);
  // map.controls[google.maps.ControlPosition.TOP_LEFT].push(destinationInput);

  // Bias the SearchBox results towards current map's viewport.
  map.addListener("bounds_changed", () => {
    searchBox.setBounds(map.getBounds());
  });
  let markers = [];
  // Listen for the event fired when the user selects a prediction and retrieve
  // more details for that place.

  searchBox.addListener("places_changed", () => {
    const places = searchBox.getPlaces();

    if (places.length == 0) {
      return;
    }
    // Clear out the old markers.
    markers.forEach((marker) => {
      marker.setMap(null);
    });
    markers = [];
    // For each place, get the icon, name and location.
    const bounds = new google.maps.LatLngBounds();
    places.forEach((place) => {
      if (!place.geometry) {
        console.log("Returned place contains no geometry");
        return;
      }
      console.log(place)
      openDrawer();
      const icon = {
        url: place.icon,
        size: new google.maps.Size(71, 71),
        origin: new google.maps.Point(0, 0),
        anchor: new google.maps.Point(17, 34),
        scaledSize: new google.maps.Size(25, 25),
      };

      // Create a marker for each place.
      markers.push(
        new google.maps.Marker({
          map,
          icon,
          title: place.name,
          position: place.geometry.location,
        })
      );
      document.getElementById("placeName").innerHTML = place.name;
      if (place.rating == undefined) {
        let displayRating = document.getElementById("ratingDisplay");
        displayRating.style.display = "none";
      } else {
        document.getElementById("rating").innerHTML = place.rating;
        document.getElementById("userReview").innerHTML =
          place.user_ratings_total;
        // Dynamically generate the rating stars based on the place's rating value
        let ratingStars = "";
        for (let i = 0; i < 5; i++) {
          if (i < Math.floor(place.rating)) {
            ratingStars += `<span>
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-6 h-6 text-yellow-400 cursor-pointer">
          <path fill-rule="evenodd" d="M10.788 3.21c.448-1.077 1.976-1.077 2.424 0l2.082 5.007 5.404.433c1.164.093 1.636 1.545.749 2.305l-4.117 3.527 1.257 5.273c.271 1.136-.964 2.033-1.96 1.425L12 18.354 7.373 21.18c-.996.608-2.231-.29-1.96-1.425l1.257-5.273-4.117-3.527c-.887-.76-.415-2.212.749-2.305l5.404-.433 2.082-5.006z"></path>
      </svg>
  </span>`;
          } else {
            ratingStars += `<span>
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-6 h-6 text-gray-300 cursor-pointer">
          <path fill-rule="evenodd" d="M10.788 3.21c.448-1.077 1.976-1.077 2.424 0l2.082 5.007 5.404.433c1.164.093 1.636 1.545.749 2.305l-4.117 3.527 1.257 5.273c.271 1.136-.964 2.033-1.96 1.425L12 18.354 7.373 21.18c-.996.608-2.231-.29-1.96-1.425l1.257-5.273-4.117-3.527c-.887-.76-.415-2.212.749-2.305l5.404-.433 2.082-5.006z"></path>
      </svg>
  </span>`;
          }
        }
        document.getElementById("ratingStars").innerHTML = ratingStars;
      }
      if (place.types.length > 1 && place.types[0] == "point_of_interest") {
        document.getElementById("placeType").innerText = "Corporate office";
      } else if (place.types[0] == null || place.types[0] == undefined) {
        document.getElementById("placeType").style.display = "none"
      }else {
        document.getElementById("placeType").innerText = place.types[0]
      }

      if (place.formatted_address == null || place.formatted_address == undefined) {
        document.getElementById("formattedAddress").style.display = "none"
      }else {
        document.getElementById("formattedAddress").innerText = place.formatted_address
      }
      
      if ( place?.current_opening_hours == undefined || place?.current_opening_hours === null) {
        document.getElementById("openHourDiv").style.display = "none"
      }else {
        document.getElementById("openingHour").innerText = "Open Now"
      }
      
      if (place.opening_hours) {
        const openingHoursDiv = document.querySelector("#dropDownDetailsList");
        openingHoursDiv.innerHTML = ""; // Clear previous content

        const openingHoursList = document.createElement("ul");

        place.opening_hours.weekday_text.forEach((period) => {
            const listItem = document.createElement("li");
            listItem.classList.add("opening-hour-list");

            listItem.textContent = period;

            openingHoursList.appendChild(listItem);
        });

        openingHoursDiv.appendChild(openingHoursList);
    } else {
        // If opening hours are not available, display a message
        document.querySelector("dropDownDetailsList").style.display = "none";
    }
    if(place.plus_code && place.plus_code !== undefined){
      document.getElementById("plusCode").innerHTML = place.plus_code.compound_code
    }else {
      document.getElementById("plusCodeDiv").style.display = "none"
    }
    if(place.formatted_phone_number && place.formatted_phone_number !== undefined){
      document.getElementById("phoneNumber").innerHTML = place.formatted_phone_number
      
    }else {
      document.getElementById("phoneNumberDiv").style.display = "none"
    }
    if(place.website && place.website !== undefined){
      document.getElementById("webisite").innerHTML = place.website
      
    }else {
      document.getElementById("webisiteDiv").style.display = "none"
    }


      if (place.geometry.viewport) {
        // Only geocodes have viewport.
        bounds.union(place.geometry.viewport);
      } else {
        bounds.extend(place.geometry.location);
      }
    });

    map.fitBounds(bounds);
  });
}

initMap();