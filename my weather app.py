import React, { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";
import axios from "axios";

const WeatherApp = () => {
  const [location, setLocation] = useState("");
  const [weather, setWeather] = useState(null);
  const [photos, setPhotos] = useState([]);
  const [feeds, setFeeds] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchWeatherAndContent = async () => {
    if (!location) return;

    setLoading(true);
    try {
      // Fetch weather data using Google services (replace with actual API endpoint)
      const weatherResponse = await axios.get(
        `https://maps.googleapis.com/maps/api/weather?key=YOUR_GOOGLE_API_KEY&query=${location}`
      );
      setWeather(weatherResponse.data);

      // Fetch photos using Google Places API (replace with actual API endpoint)
      const photoResponse = await axios.get(
        `https://maps.googleapis.com/maps/api/place/photo?key=YOUR_GOOGLE_API_KEY&query=${location}`
      );
      setPhotos(photoResponse.data.results);

      // Fetch news feeds related to the location (replace with a real API)
      const feedsResponse = await axios.get(
        `https://newsapi.org/v2/everything?q=${location}&apiKey=YOUR_NEWS_API_KEY`
      );
      setFeeds(feedsResponse.data.articles);
    } catch (error) {
      console.error("Error fetching data:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const fetchUserLocation = async () => {
      try {
        const locationResponse = await axios.get(
          `https://www.googleapis.com/geolocation/v1/geolocate?key=YOUR_GOOGLE_API_KEY`
        );
        const { lat, lng } = locationResponse.data.location;
        setLocation(`${lat},${lng}`);
        fetchWeatherAndContent();
      } catch (error) {
        console.error("Error fetching user location:", error);
      }
    };

    fetchUserLocation();
  }, []);

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-4 text-center">South Africa Weather App</h1>
        <div className="flex items-center space-x-2 mb-4">
          <Input
            placeholder="Enter a location in South Africa"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
          />
          <Button onClick={fetchWeatherAndContent} disabled={loading}>
            <Search className="mr-2 h-4 w-4" /> Search
          </Button>
        </div>
        {loading && <p className="text-center">Loading...</p>}
        {weather && (
          <Card className="mb-4">
            <CardContent>
              <h2 className="text-xl font-bold">Weather in {weather.location.name}</h2>
              <p>Temperature: {weather.current.temp_c}°C</p>
              <p>Condition: {weather.current.condition.text}</p>
              <img
                src={weather.current.condition.icon}
                alt={weather.current.condition.text}
              />
            </CardContent>
          </Card>
        )}
        {photos.length > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {photos.map((photo) => (
              <img
                key={photo.id}
                src={photo.urls.small}
                alt={photo.alt_description}
                className="rounded-xl shadow-md"
              />
            ))}
          </div>
        )}
        {feeds.length > 0 && (
          <div className="mt-4">
            <h2 className="text-2xl font-bold mb-2">Latest News Feeds</h2>
            <ul className="space-y-2">
              {feeds.map((feed, index) => (
                <li key={index} className="border p-2 rounded-md shadow-sm">
                  <a
                    href={feed.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:underline"
                  >
                    {feed.title}
                  </a>
                  <p className="text-sm text-gray-600">{feed.description}</p>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default WeatherApp;
import axios from 'axios';

// Replace with your own OpenWeatherMap API key
const API_KEY = 'your-api-key';
const BASE_URL = 'https://api.openweathermap.org/data/2.5/';

export const getWeather = async (city) => {
  try {
    const response = await axios.get(`${BASE_URL}weather`, {
      params: {
        q: city,
        appid: API_KEY,
        units: 'metric', // You can change this to 'imperial' for Fahrenheit
      },
    });
    return response.data;
  } catch (error) {
    console.error(error);
    return null;
  }
};

export const getForecast = async (city) => {
  try {
    const response = await axios.get(`${BASE_URL}forecast`, {
      params: {
        q: city,
        appid: API_KEY,
        units: 'metric',
      },
    });
    return response.data;
  } catch (error) {
    console.error(error);
    return null;
  }
};
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createDrawerNavigator } from '@react-navigation/drawer';
import WeatherScreen from './screens/WeatherScreen';
import SportsScreen from './screens/SportsScreen';
import ShopsScreen from './screens/ShopsScreen';
import SchoolsScreen from './screens/SchoolsScreen';
import PlacesScreen from './screens/PlacesScreen';

const Drawer = createDrawerNavigator();

const App = () => {
  return (
    <NavigationContainer>
      <Drawer.Navigator initialRouteName="Weather">
        <Drawer.Screen name="Weather" component={WeatherScreen} />
        <Drawer.Screen name="Sports" component={SportsScreen} />
        <Drawer.Screen name="Shops" component={ShopsScreen} />
        <Drawer.Screen name="Schools" component={SchoolsScreen} />
        <Drawer.Screen name="Places" component={PlacesScreen} />
      </Drawer.Navigator>
    </NavigationContainer>
  );
};

export default App;
import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, StyleSheet } from 'react-native';

const SportsScreen = () => {
  const [sportsData, setSportsData] = useState(null);

  useEffect(() => {
    // Fetch sports data from an API like SportsRadar, ESPN, etc.
    // For simplicity, using mock data here
    setSportsData({
      football: ['Match 1: Team A vs Team B', 'Match 2: Team C vs Team D'],
      rugby: ['Match 1: Team X vs Team Y'],
      cricket: ['Match 1: Team 1 vs Team 2'],
    });
  }, []);

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.header}>Live Sports Scores</Text>
      {sportsData && (
        <View>
          <Text>Football:</Text>
          {sportsData.football.map((match, index) => (
            <Text key={index}>{match}</Text>
          ))}
          <Text>Rugby:</Text>
          {sportsData.rugby.map((match, index) => (
            <Text key={index}>{match}</Text>
          ))}
          <Text>Cricket:</Text>
          {sportsData.cricket.map((match, index) => (
            <Text key={index}>{match}</Text>
          ))}
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    padding: 20,
  },
  header: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
  },
});

export default SportsScreen;
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import MapView, { Marker } from 'react-native-maps';

const PlacesScreen = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.header}>Places Around South Africa</Text>
      <MapView
        style={styles.map}
        initialRegion={{
          latitude: -30.5595, // South Africa's approximate center
          longitude: 22.9375,
          latitudeDelta: 5,
          longitudeDelta: 5,
        }}
      >
        <Marker coordinate={{ latitude: -33.9249, longitude: 18.4241 }} title="Cape Town" />
        {/* Add more markers for other places */}
      </MapView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  map: {
    flex: 1,
  },
});

export default PlacesScreen;
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import MapView, { Marker } from 'react-native-maps';

const PlacesScreen = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.header}>Places Around South Africa</Text>
      <MapView
        style={styles.map}
        initialRegion={{
          latitude: -30.5595, // South Africa's approximate center
          longitude: 22.9375,
          latitudeDelta: 5,
          longitudeDelta: 5,
        }}
      >
        <Marker coordinate={{ latitude: -33.9249, longitude: 18.4241 }} title="Cape Town" />
        {/* Add more markers for other places */}
      </MapView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
  },
  map: {
    flex: 1,
  },
});

export default PlacesScreen;
