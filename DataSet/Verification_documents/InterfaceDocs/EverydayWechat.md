# API Documentation

## Function: `get_dictum_info`

### Description
Fetches a daily quote based on the specified channel.

### Parameters
- **channel** (`int`): The channel identifier for the quote source. It should be an integer corresponding to a key in the `DICTUM_NAME_DICT`.
  - **Value Range**: 1 to 7 (inclusive)

### Returns
- **str**: A string containing the daily quote. Returns `None` if the channel is invalid or not provided.

### Purpose
This function retrieves a daily quote from a specified source based on the provided channel identifier.

---

## Function: `get_weather_info`

### Description
Fetches the weather information for a specified city.

### Parameters
- **cityname** (`str`): The name of the city for which the weather information is requested.
- **is_tomorrow** (`bool`, optional): A flag indicating whether to fetch the weather for tomorrow. Defaults to `False`.

### Returns
- **str**: A string describing the weather conditions. Returns `None` if the city name is not provided.

### Purpose
This function retrieves current or tomorrow's weather information for a specified city.

---

## Function: `get_bot_info`

### Description
Interacts with an AI chatbot to get a response based on the user's message.

### Parameters
- **message** (`str`): The message sent to the chatbot.
- **userId** (`str`, optional): A unique identifier for the user, used to track the conversation. Defaults to an empty string.

### Returns
- **str**: The chatbot's response to the user's message. Returns `None` if the bot fails to respond.

### Purpose
This function facilitates interaction with an AI chatbot, allowing users to receive automated replies based on their input.

---

## Function: `get_diff_time`

### Description
Calculates the number of days since a specified start date.

### Parameters
- **start_date** (`str`): The start date in the format "YYYY-MM-DD".
- **start_msg** (`str`, optional): A message template that can include a placeholder `{}` for the number of days. Defaults to an empty string.

### Returns
- **str**: A message indicating the number of days since the start date. Returns `None` if the date format is invalid.

### Purpose
This function calculates and returns a message indicating how many days have passed since a specified date.

---

## Function: `get_constellation_info`

### Description
Fetches horoscope information based on a given birthday or constellation name.

### Parameters
- **birthday_str** (`str`): A string representing either a birthday in the format "MM-DD" or "YYYY-MM-DD", or the name of a constellation.
- **is_tomorrow** (`bool`, optional): A flag indicating whether to fetch tomorrow's horoscope. Defaults to `False`.

### Returns
- **str**: A string containing the horoscope information. Returns `None` if the input is invalid.

### Purpose
This function retrieves horoscope information based on the user's birthday or constellation name.

---

## Function: `get_calendar_info`

### Description
Fetches calendar information for today or tomorrow.

### Parameters
- **calendar** (`bool`, optional): A flag indicating whether to fetch calendar information. Defaults to `True`.
- **is_tomorrow** (`bool`, optional): A flag indicating whether to fetch information for tomorrow. Defaults to `False`.
- **_date** (`str`, optional): A specific date in the format "YYYYMMDD". Defaults to an empty string.

### Returns
- **str**: A string containing the calendar information. Returns `None` if calendar fetching is disabled or if the date is invalid.

### Purpose
This function retrieves calendar information, including date and relevant details, for today or tomorrow based on the provided parameters.

