library(sf)
library(ggplot2)
library(tigris)
library(httr)
library(jsonlite)
library(dplyr)

epa_stations <- data.frame(
  station_name = c(
    "Lancaster",
    "Reseda",
    "Pasadena",
    "Azusa",
    "Central City",
    "Pico Rivera",
    "Compton",
    "Signal Hill",
    "Long Beach (Route 710 Near Road)",
    "Long Beach (South)",
    "Long Beach (North)"
  ),
  longitude = c(
    -118.1472, -118.5376, -118.1486, -117.8986, -118.2458,
    -118.0964, -118.2201, -118.1675, -118.2025, -118.1592, -118.1892
  ),
  latitude = c(
    34.6969, 34.2104, 34.1631, 34.1264, 34.0404,
    33.9983, 33.8958, 33.7975, 33.8225, 33.7684, 33.7978
  ),
  district = c(
    5,  # Lancaster - District 5
    3,  # Reseda - District 3
    5,  # Pasadena - District 5
    5,  # Azusa - District 5
    1,  # Central City - District 1
    2,  # Pico Rivera - District 2
    2,  # Compton - District 2
    4,  # Signal Hill - District 4
    4,  # Long Beach (Route 710) - District 4
    4,  # Long Beach (South) - District 4
    4   # Long Beach (North) - District 4
  ),
  stringsAsFactors = FALSE
)

# Convert to sf object
epa_stations_sf <- st_as_sf(epa_stations, 
                            coords = c("longitude", "latitude"), 
                            crs = 4326)

# Get LA County
la_county <- counties(state = "CA", cb = TRUE, class = "sf") %>%
  filter(NAME == "Los Angeles")

# Get Supervisorial Districts
url_districts <- "https://services9.arcgis.com/2ynJbr9BE17vXxR8/ArcGIS/rest/services/Supervisorial_Districts_(Current)/FeatureServer/0/query"

params_districts <- list(
  where = "1=1",
  outFields = "*",
  f = "geojson"
)

response <- GET(url_districts, query = params_districts)
districts_geojson <- content(response, as = "text", encoding = "UTF-8")
districts_sf <- st_read(districts_geojson, quiet = TRUE)
districts_sf <- st_transform(districts_sf, st_crs(la_county))


# Get centroids for district labels
district_centroids <- st_centroid(districts_sf)

# Define distinct colors for each district
district_colors <- c(
  "1" = "#E41A1C",  # Red
  "2" = "#377EB8",  # Blue
  "3" = "#4DAF4A",  # Green
  "4" = "#984EA3",  # Purple
  "5" = "#FF7F00"   # Orange
)

# Create the figure
p2 <- ggplot() +
  # County boundary
  geom_sf(data = la_county, 
          fill = "gray98", 
          color = "black", 
          size = 0.8) +
  
  # Supervisorial Districts with distinct colors
  geom_sf(data = districts_sf, 
          aes(fill = as.factor(DISTRICT)), 
          color = "black", 
          size = 0.5, 
          alpha = 0.2) +
  
  # District labels with matching colors
  geom_sf_text(data = district_centroids, 
               aes(label = paste("District", DISTRICT),
                   color = as.factor(DISTRICT)),
               size = 4.5,
               fontface = "bold",
               show.legend = FALSE) +
  
  # EPA Stations - BLACK DOTS (no labels)
  geom_sf(data = epa_stations_sf,
          aes(color = "PM2.5 Monitoring Stations"),
          size = 3.5,
          shape = 16) +
  
  # Color scales
  scale_fill_manual(
    values = district_colors,
    name = "Supervisorial District",
    labels = c("District 1", "District 2", "District 3", "District 4", "District 5"),
    guide = guide_legend(override.aes = list(color = district_colors, 
                                             fill = district_colors))
  ) +
  
  scale_color_manual(
    values = c("PM2.5 Monitoring Stations" = "black"),
    name = NULL,  # Remove the title for this legend
    guide = guide_legend(override.aes = list(size = 3.5, shape = 16))
  ) +
  
  # Clean theme
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    axis.text = element_blank(),
    axis.title = element_blank(),
    axis.ticks = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16, margin = margin(b = 8)),
    plot.subtitle = element_text(hjust = 0.5, size = 12, color = "gray40", margin = margin(b = 12)),
    plot.caption = element_text(hjust = 0.5, size = 10, color = "gray50", margin = margin(t = 10)),
    legend.position = "right",
    legend.title = element_text(face = "bold", size = 11),
    legend.text = element_text(size = 10),
    legend.spacing.y = unit(0.5, "cm"),
    plot.margin = margin(20, 20, 20, 20)
  ) 

# Display the figure
print(p2)
