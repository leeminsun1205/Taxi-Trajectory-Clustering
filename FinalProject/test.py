import json
import csv

def convert_json_to_csv(json_file_path, csv_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f_json:
        full_json_data = json.load(f_json)

    city_id = full_json_data.get('city_id')
    city_name = full_json_data.get('city_name')
    country_code = full_json_data.get('country_code')
    
    data_entries = full_json_data.get('data', [])

    headers = ['city_id', 'city_name', 'country_code']
    
    if data_entries:
        first_entry_keys = [key for key in data_entries[0].keys() if key != 'weather']
        headers.extend(first_entry_keys)
    
    # Add specific keys for the nested 'weather' object
    headers.extend(['weather_description', 'weather_icon', 'weather_code'])

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(headers)

        for entry in data_entries:
            row = []
            # Add common city data
            row.append(city_id)
            row.append(city_name)
            row.append(country_code)
            
            if data_entries:
                for key in first_entry_keys:
                    row.append(entry.get(key)) 
            
            weather_data = entry.get('weather', {}) 
            row.append(weather_data.get('description'))
            row.append(weather_data.get('icon'))
            row.append(weather_data.get('code'))
            
            writer.writerow(row)

input_json_file = 'hcmcApril.json' 
output_csv_file = 'output_data.csv'

"""
{
  "city_id": "1566083",
  "city_name": "Ho Chi Minh City",
  "country_code": "VN",
  "data": [
    {
      "app_temp": 30.5, "azimuth": 88.3, "clouds": 87, "dewpt": 24.8, "dhi": 69,
      "dni": 614, "elev_angle": 16.2, "ghi": 234, "pod": "d", "precip_rate": 0,
      "pres": 1007, "revision_status": "final", "rh": 88, "slp": 1009,
      "snow_rate": 0, "solar_rad": 85, "temp": 27,
      "timestamp_local": "2025-04-01T07:00:00",
      "timestamp_utc": "2025-04-01T00:00:00", "ts": 1743465600, "uv": 0.9,
      "vis": 5, "weather": {"description": "Overcast clouds", "icon": "c04d", "code": 804},
      "wind_dir": 290, "wind_gust_spd": 5.2, "wind_spd": 1.5
    },
    {
      "app_temp": 31.7, "azimuth": 89, "clouds": 68, "dewpt": 24.9, "dhi": 76,
      "dni": 658, "elev_angle": 19.9, "ghi": 299, "pod": "d", "precip_rate": 0,
      "pres": 1007, "revision_status": "final", "rh": 86, "slp": 1009,
      "snow_rate": 0, "solar_rad": 204, "temp": 27.5,
      "timestamp_local": "2025-04-01T07:15:00",
      "timestamp_utc": "2025-04-01T00:15:00", "ts": 1743466500, "uv": 1.3,
      "vis": 5, "weather": {"description": "Broken clouds", "icon": "c03d", "code": 803},
      "wind_dir": 280, "wind_gust_spd": 5.7, "wind_spd": 1.5
    }
  ]
}
"""
convert_json_to_csv(input_json_file, output_csv_file)

print(f"Conversion complete. Data from '{input_json_file}' written to '{output_csv_file}'.")