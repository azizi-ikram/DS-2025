```python
# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
kundanbedmutha_hotel_booking_reservation_path = kagglehub.dataset_download('kundanbedmutha/hotel-booking-reservation')

print('Data source import complete.')

```

    Downloading from https://www.kaggle.com/api/v1/datasets/download/kundanbedmutha/hotel-booking-reservation?dataset_version_number=1...


    100%|██████████| 3.48M/3.48M [00:00<00:00, 62.5MB/s]

    Extracting files...


    


    Data source import complete.


# Introduction
This dataset contains detailed information about hotel bookings, including reservation dates, customer demographics, stay duration, deposit types, cancellation status, and other booking-related attributes. It is an updated and cleaned version of the original UCI “Hotel Booking Demand” dataset, prepared for machine learning and data analytics tasks.


```python
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
```


```python
import pandas as pd
df=pd.read_csv(f'{kundanbedmutha_hotel_booking_reservation_path}/hotel_bookings_updated_2024.csv')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 119390 entries, 0 to 119389
    Data columns (total 33 columns):
     #   Column                          Non-Null Count   Dtype  
    ---  ------                          --------------   -----  
     0   hotel                           119390 non-null  object 
     1   is_canceled                     119390 non-null  int64  
     2   lead_time                       119390 non-null  int64  
     3   arrival_date_year               119390 non-null  int64  
     4   arrival_date_month              119390 non-null  object 
     5   arrival_date_week_number        119390 non-null  int64  
     6   arrival_date_day_of_month       119390 non-null  int64  
     7   stays_in_weekend_nights         119390 non-null  int64  
     8   stays_in_week_nights            119390 non-null  int64  
     9   adults                          119390 non-null  int64  
     10  children                        119386 non-null  float64
     11  babies                          119390 non-null  int64  
     12  meal                            119390 non-null  object 
     13  country                         118902 non-null  object 
     14  market_segment                  119390 non-null  object 
     15  distribution_channel            119390 non-null  object 
     16  is_repeated_guest               119390 non-null  int64  
     17  previous_cancellations          119390 non-null  int64  
     18  previous_bookings_not_canceled  119390 non-null  int64  
     19  reserved_room_type              119390 non-null  object 
     20  assigned_room_type              119390 non-null  object 
     21  booking_changes                 119390 non-null  int64  
     22  deposit_type                    119390 non-null  object 
     23  agent                           103050 non-null  float64
     24  company                         6797 non-null    float64
     25  days_in_waiting_list            119390 non-null  int64  
     26  customer_type                   119390 non-null  object 
     27  adr                             119390 non-null  float64
     28  required_car_parking_spaces     119390 non-null  int64  
     29  total_of_special_requests       119390 non-null  int64  
     30  reservation_status              119390 non-null  object 
     31  reservation_status_date         119390 non-null  object 
     32  city                            119390 non-null  object 
    dtypes: float64(4), int64(16), object(13)
    memory usage: 30.1+ MB



```python
df.isna().sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hotel</th>
      <td>0</td>
    </tr>
    <tr>
      <th>is_canceled</th>
      <td>0</td>
    </tr>
    <tr>
      <th>lead_time</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_year</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_month</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_week_number</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_day_of_month</th>
      <td>0</td>
    </tr>
    <tr>
      <th>stays_in_weekend_nights</th>
      <td>0</td>
    </tr>
    <tr>
      <th>stays_in_week_nights</th>
      <td>0</td>
    </tr>
    <tr>
      <th>adults</th>
      <td>0</td>
    </tr>
    <tr>
      <th>children</th>
      <td>4</td>
    </tr>
    <tr>
      <th>babies</th>
      <td>0</td>
    </tr>
    <tr>
      <th>meal</th>
      <td>0</td>
    </tr>
    <tr>
      <th>country</th>
      <td>488</td>
    </tr>
    <tr>
      <th>market_segment</th>
      <td>0</td>
    </tr>
    <tr>
      <th>distribution_channel</th>
      <td>0</td>
    </tr>
    <tr>
      <th>is_repeated_guest</th>
      <td>0</td>
    </tr>
    <tr>
      <th>previous_cancellations</th>
      <td>0</td>
    </tr>
    <tr>
      <th>previous_bookings_not_canceled</th>
      <td>0</td>
    </tr>
    <tr>
      <th>reserved_room_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>assigned_room_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>booking_changes</th>
      <td>0</td>
    </tr>
    <tr>
      <th>deposit_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>agent</th>
      <td>16340</td>
    </tr>
    <tr>
      <th>company</th>
      <td>112593</td>
    </tr>
    <tr>
      <th>days_in_waiting_list</th>
      <td>0</td>
    </tr>
    <tr>
      <th>customer_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>adr</th>
      <td>0</td>
    </tr>
    <tr>
      <th>required_car_parking_spaces</th>
      <td>0</td>
    </tr>
    <tr>
      <th>total_of_special_requests</th>
      <td>0</td>
    </tr>
    <tr>
      <th>reservation_status</th>
      <td>0</td>
    </tr>
    <tr>
      <th>reservation_status_date</th>
      <td>0</td>
    </tr>
    <tr>
      <th>city</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
df.head()
```





  <div id="df-0b73482c-5135-4ea6-b0f3-cdd7ab9b10c3" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>...</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Resort Hotel - Chandigarh</td>
      <td>0</td>
      <td>342</td>
      <td>2024</td>
      <td>July</td>
      <td>30</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2024-07-27 22:16:40.916332324</td>
      <td>Chandigarh</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Resort Hotel - Mumbai</td>
      <td>0</td>
      <td>737</td>
      <td>2024</td>
      <td>April</td>
      <td>17</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2024-04-28 21:56:21.507509066</td>
      <td>Mumbai</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Resort Hotel - Delhi</td>
      <td>0</td>
      <td>7</td>
      <td>2024</td>
      <td>September</td>
      <td>37</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2024-09-10 03:46:25.734029096</td>
      <td>Delhi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Resort Hotel - Kolkata</td>
      <td>0</td>
      <td>13</td>
      <td>2024</td>
      <td>August</td>
      <td>33</td>
      <td>14</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>304.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>75.0</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2024-08-14 18:07:10.049669568</td>
      <td>Kolkata</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Resort Hotel - Lucknow</td>
      <td>0</td>
      <td>14</td>
      <td>2024</td>
      <td>September</td>
      <td>37</td>
      <td>14</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>240.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>98.0</td>
      <td>0</td>
      <td>1</td>
      <td>Check-Out</td>
      <td>2024-09-14 14:27:32.473846000</td>
      <td>Lucknow</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-0b73482c-5135-4ea6-b0f3-cdd7ab9b10c3')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-0b73482c-5135-4ea6-b0f3-cdd7ab9b10c3 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0b73482c-5135-4ea6-b0f3-cdd7ab9b10c3');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-27d41797-2a33-43bc-ab6d-d4318d80344b">
      <button class="colab-df-quickchart" onclick="quickchart('df-27d41797-2a33-43bc-ab6d-d4318d80344b')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-27d41797-2a33-43bc-ab6d-d4318d80344b button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
df.tail()
```





  <div id="df-bbe3dee2-a487-4b13-a8a6-0154c58cd811" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hotel</th>
      <th>is_canceled</th>
      <th>lead_time</th>
      <th>arrival_date_year</th>
      <th>arrival_date_month</th>
      <th>arrival_date_week_number</th>
      <th>arrival_date_day_of_month</th>
      <th>stays_in_weekend_nights</th>
      <th>stays_in_week_nights</th>
      <th>adults</th>
      <th>...</th>
      <th>agent</th>
      <th>company</th>
      <th>days_in_waiting_list</th>
      <th>customer_type</th>
      <th>adr</th>
      <th>required_car_parking_spaces</th>
      <th>total_of_special_requests</th>
      <th>reservation_status</th>
      <th>reservation_status_date</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>119385</th>
      <td>City Hotel - Pune</td>
      <td>0</td>
      <td>23</td>
      <td>2024</td>
      <td>September</td>
      <td>39</td>
      <td>29</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>...</td>
      <td>394.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>96.14</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2024-09-29 05:33:06.002060492</td>
      <td>Pune</td>
    </tr>
    <tr>
      <th>119386</th>
      <td>City Hotel - Mumbai</td>
      <td>0</td>
      <td>102</td>
      <td>2024</td>
      <td>November</td>
      <td>46</td>
      <td>16</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>...</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>225.43</td>
      <td>0</td>
      <td>2</td>
      <td>Check-Out</td>
      <td>2024-11-16 01:55:18.426320680</td>
      <td>Mumbai</td>
    </tr>
    <tr>
      <th>119387</th>
      <td>City Hotel - Lucknow</td>
      <td>0</td>
      <td>34</td>
      <td>2024</td>
      <td>April</td>
      <td>16</td>
      <td>19</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>...</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>157.71</td>
      <td>0</td>
      <td>4</td>
      <td>Check-Out</td>
      <td>2024-04-19 07:50:22.982016768</td>
      <td>Lucknow</td>
    </tr>
    <tr>
      <th>119388</th>
      <td>City Hotel - Ahmedabad</td>
      <td>0</td>
      <td>109</td>
      <td>2024</td>
      <td>October</td>
      <td>40</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>...</td>
      <td>89.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>104.40</td>
      <td>0</td>
      <td>0</td>
      <td>Check-Out</td>
      <td>2024-10-05 12:37:40.429352788</td>
      <td>Ahmedabad</td>
    </tr>
    <tr>
      <th>119389</th>
      <td>City Hotel - Bhopal</td>
      <td>0</td>
      <td>205</td>
      <td>2024</td>
      <td>December</td>
      <td>51</td>
      <td>21</td>
      <td>2</td>
      <td>7</td>
      <td>2</td>
      <td>...</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>0</td>
      <td>Transient</td>
      <td>151.20</td>
      <td>0</td>
      <td>2</td>
      <td>Check-Out</td>
      <td>2024-12-21 07:11:08.111802592</td>
      <td>Bhopal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-bbe3dee2-a487-4b13-a8a6-0154c58cd811')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-bbe3dee2-a487-4b13-a8a6-0154c58cd811 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-bbe3dee2-a487-4b13-a8a6-0154c58cd811');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-5afb7f43-39d7-44df-8cfb-b429f4806854">
      <button class="colab-df-quickchart" onclick="quickchart('df-5afb7f43-39d7-44df-8cfb-b429f4806854')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-5afb7f43-39d7-44df-8cfb-b429f4806854 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




# Cleaning the Data and filling the null values


```python
df['city'].value_counts()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>city</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bhopal</th>
      <td>8104</td>
    </tr>
    <tr>
      <th>Jaipur</th>
      <td>8038</td>
    </tr>
    <tr>
      <th>Ahmedabad</th>
      <td>8028</td>
    </tr>
    <tr>
      <th>Pune</th>
      <td>7992</td>
    </tr>
    <tr>
      <th>Hyderabad</th>
      <td>7981</td>
    </tr>
    <tr>
      <th>Delhi</th>
      <td>7978</td>
    </tr>
    <tr>
      <th>Chandigarh</th>
      <td>7978</td>
    </tr>
    <tr>
      <th>Kolkata</th>
      <td>7976</td>
    </tr>
    <tr>
      <th>Goa</th>
      <td>7973</td>
    </tr>
    <tr>
      <th>Mumbai</th>
      <td>7935</td>
    </tr>
    <tr>
      <th>Chennai</th>
      <td>7925</td>
    </tr>
    <tr>
      <th>Bangalore</th>
      <td>7897</td>
    </tr>
    <tr>
      <th>Kochi</th>
      <td>7889</td>
    </tr>
    <tr>
      <th>Lucknow</th>
      <td>7869</td>
    </tr>
    <tr>
      <th>Indore</th>
      <td>7827</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python


india_cities = ['Bhopal', 'Jaipur', 'Ahmedabad', 'Pune', 'Hyderabad',
                'Chandigarh', 'Delhi', 'Kolkata', 'Goa', 'Mumbai',
                'Chennai', 'Bangalore', 'Kochi', 'Lucknow', 'Indore']

df['country'] = df['country'].mask(
    df['city'].isin(india_cities) & df['country'].isna(),
    'India'
)
```


```python
df.isna().sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hotel</th>
      <td>0</td>
    </tr>
    <tr>
      <th>is_canceled</th>
      <td>0</td>
    </tr>
    <tr>
      <th>lead_time</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_year</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_month</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_week_number</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_day_of_month</th>
      <td>0</td>
    </tr>
    <tr>
      <th>stays_in_weekend_nights</th>
      <td>0</td>
    </tr>
    <tr>
      <th>stays_in_week_nights</th>
      <td>0</td>
    </tr>
    <tr>
      <th>adults</th>
      <td>0</td>
    </tr>
    <tr>
      <th>children</th>
      <td>4</td>
    </tr>
    <tr>
      <th>babies</th>
      <td>0</td>
    </tr>
    <tr>
      <th>meal</th>
      <td>0</td>
    </tr>
    <tr>
      <th>country</th>
      <td>0</td>
    </tr>
    <tr>
      <th>market_segment</th>
      <td>0</td>
    </tr>
    <tr>
      <th>distribution_channel</th>
      <td>0</td>
    </tr>
    <tr>
      <th>is_repeated_guest</th>
      <td>0</td>
    </tr>
    <tr>
      <th>previous_cancellations</th>
      <td>0</td>
    </tr>
    <tr>
      <th>previous_bookings_not_canceled</th>
      <td>0</td>
    </tr>
    <tr>
      <th>reserved_room_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>assigned_room_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>booking_changes</th>
      <td>0</td>
    </tr>
    <tr>
      <th>deposit_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>agent</th>
      <td>16340</td>
    </tr>
    <tr>
      <th>company</th>
      <td>112593</td>
    </tr>
    <tr>
      <th>days_in_waiting_list</th>
      <td>0</td>
    </tr>
    <tr>
      <th>customer_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>adr</th>
      <td>0</td>
    </tr>
    <tr>
      <th>required_car_parking_spaces</th>
      <td>0</td>
    </tr>
    <tr>
      <th>total_of_special_requests</th>
      <td>0</td>
    </tr>
    <tr>
      <th>reservation_status</th>
      <td>0</td>
    </tr>
    <tr>
      <th>reservation_status_date</th>
      <td>0</td>
    </tr>
    <tr>
      <th>city</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
most_frequent_agent = df['agent'].mode()
if not most_frequent_agent.empty:
    df['agent'] = df['agent'].fillna(most_frequent_agent[0])
else:
    df['agent'] = df['agent'].fillna(0)

df.isna().sum()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hotel</th>
      <td>0</td>
    </tr>
    <tr>
      <th>is_canceled</th>
      <td>0</td>
    </tr>
    <tr>
      <th>lead_time</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_year</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_month</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_week_number</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_day_of_month</th>
      <td>0</td>
    </tr>
    <tr>
      <th>stays_in_weekend_nights</th>
      <td>0</td>
    </tr>
    <tr>
      <th>stays_in_week_nights</th>
      <td>0</td>
    </tr>
    <tr>
      <th>adults</th>
      <td>0</td>
    </tr>
    <tr>
      <th>children</th>
      <td>4</td>
    </tr>
    <tr>
      <th>babies</th>
      <td>0</td>
    </tr>
    <tr>
      <th>meal</th>
      <td>0</td>
    </tr>
    <tr>
      <th>country</th>
      <td>0</td>
    </tr>
    <tr>
      <th>market_segment</th>
      <td>0</td>
    </tr>
    <tr>
      <th>distribution_channel</th>
      <td>0</td>
    </tr>
    <tr>
      <th>is_repeated_guest</th>
      <td>0</td>
    </tr>
    <tr>
      <th>previous_cancellations</th>
      <td>0</td>
    </tr>
    <tr>
      <th>previous_bookings_not_canceled</th>
      <td>0</td>
    </tr>
    <tr>
      <th>reserved_room_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>assigned_room_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>booking_changes</th>
      <td>0</td>
    </tr>
    <tr>
      <th>deposit_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>agent</th>
      <td>0</td>
    </tr>
    <tr>
      <th>company</th>
      <td>112593</td>
    </tr>
    <tr>
      <th>days_in_waiting_list</th>
      <td>0</td>
    </tr>
    <tr>
      <th>customer_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>adr</th>
      <td>0</td>
    </tr>
    <tr>
      <th>required_car_parking_spaces</th>
      <td>0</td>
    </tr>
    <tr>
      <th>total_of_special_requests</th>
      <td>0</td>
    </tr>
    <tr>
      <th>reservation_status</th>
      <td>0</td>
    </tr>
    <tr>
      <th>reservation_status_date</th>
      <td>0</td>
    </tr>
    <tr>
      <th>city</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
df['company'].fillna('Unkowns', inplace=True)
df.isna().sum()
```

    /tmp/ipython-input-1744419566.py:1: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      df['company'].fillna('Unkowns', inplace=True)
    /tmp/ipython-input-1744419566.py:1: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value 'Unkowns' has dtype incompatible with float64, please explicitly cast to a compatible dtype first.
      df['company'].fillna('Unkowns', inplace=True)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hotel</th>
      <td>0</td>
    </tr>
    <tr>
      <th>is_canceled</th>
      <td>0</td>
    </tr>
    <tr>
      <th>lead_time</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_year</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_month</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_week_number</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_day_of_month</th>
      <td>0</td>
    </tr>
    <tr>
      <th>stays_in_weekend_nights</th>
      <td>0</td>
    </tr>
    <tr>
      <th>stays_in_week_nights</th>
      <td>0</td>
    </tr>
    <tr>
      <th>adults</th>
      <td>0</td>
    </tr>
    <tr>
      <th>children</th>
      <td>4</td>
    </tr>
    <tr>
      <th>babies</th>
      <td>0</td>
    </tr>
    <tr>
      <th>meal</th>
      <td>0</td>
    </tr>
    <tr>
      <th>country</th>
      <td>0</td>
    </tr>
    <tr>
      <th>market_segment</th>
      <td>0</td>
    </tr>
    <tr>
      <th>distribution_channel</th>
      <td>0</td>
    </tr>
    <tr>
      <th>is_repeated_guest</th>
      <td>0</td>
    </tr>
    <tr>
      <th>previous_cancellations</th>
      <td>0</td>
    </tr>
    <tr>
      <th>previous_bookings_not_canceled</th>
      <td>0</td>
    </tr>
    <tr>
      <th>reserved_room_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>assigned_room_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>booking_changes</th>
      <td>0</td>
    </tr>
    <tr>
      <th>deposit_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>agent</th>
      <td>0</td>
    </tr>
    <tr>
      <th>company</th>
      <td>0</td>
    </tr>
    <tr>
      <th>days_in_waiting_list</th>
      <td>0</td>
    </tr>
    <tr>
      <th>customer_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>adr</th>
      <td>0</td>
    </tr>
    <tr>
      <th>required_car_parking_spaces</th>
      <td>0</td>
    </tr>
    <tr>
      <th>total_of_special_requests</th>
      <td>0</td>
    </tr>
    <tr>
      <th>reservation_status</th>
      <td>0</td>
    </tr>
    <tr>
      <th>reservation_status_date</th>
      <td>0</td>
    </tr>
    <tr>
      <th>city</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
df['children'].fillna(0, inplace=True)
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
df.isna().sum()
```

    /tmp/ipython-input-3258936006.py:1: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      df['children'].fillna(0, inplace=True)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hotel</th>
      <td>0</td>
    </tr>
    <tr>
      <th>is_canceled</th>
      <td>0</td>
    </tr>
    <tr>
      <th>lead_time</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_year</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_month</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_week_number</th>
      <td>0</td>
    </tr>
    <tr>
      <th>arrival_date_day_of_month</th>
      <td>0</td>
    </tr>
    <tr>
      <th>stays_in_weekend_nights</th>
      <td>0</td>
    </tr>
    <tr>
      <th>stays_in_week_nights</th>
      <td>0</td>
    </tr>
    <tr>
      <th>adults</th>
      <td>0</td>
    </tr>
    <tr>
      <th>children</th>
      <td>0</td>
    </tr>
    <tr>
      <th>babies</th>
      <td>0</td>
    </tr>
    <tr>
      <th>meal</th>
      <td>0</td>
    </tr>
    <tr>
      <th>country</th>
      <td>0</td>
    </tr>
    <tr>
      <th>market_segment</th>
      <td>0</td>
    </tr>
    <tr>
      <th>distribution_channel</th>
      <td>0</td>
    </tr>
    <tr>
      <th>is_repeated_guest</th>
      <td>0</td>
    </tr>
    <tr>
      <th>previous_cancellations</th>
      <td>0</td>
    </tr>
    <tr>
      <th>previous_bookings_not_canceled</th>
      <td>0</td>
    </tr>
    <tr>
      <th>reserved_room_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>assigned_room_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>booking_changes</th>
      <td>0</td>
    </tr>
    <tr>
      <th>deposit_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>agent</th>
      <td>0</td>
    </tr>
    <tr>
      <th>company</th>
      <td>0</td>
    </tr>
    <tr>
      <th>days_in_waiting_list</th>
      <td>0</td>
    </tr>
    <tr>
      <th>customer_type</th>
      <td>0</td>
    </tr>
    <tr>
      <th>adr</th>
      <td>0</td>
    </tr>
    <tr>
      <th>required_car_parking_spaces</th>
      <td>0</td>
    </tr>
    <tr>
      <th>total_of_special_requests</th>
      <td>0</td>
    </tr>
    <tr>
      <th>reservation_status</th>
      <td>0</td>
    </tr>
    <tr>
      <th>reservation_status_date</th>
      <td>0</td>
    </tr>
    <tr>
      <th>city</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>




```python
df.duplicated().sum()
```




    np.int64(0)



# Overall Booking Cancellation Rate
The overall booking cancellation rate stands at **37%**, meaning nearly one in every three reservations is lost before arrival—a critically high figure for the Indian hotel market. This translates into massive revenue leakage and unreliable occupancy, creating a strong business case for immediate countermeasures such as **25–30% strategic overbooking**, wider adoption of **non-refundable rates with deposits**, and stricter cancellation penalties to protect revenue and improve forecasting accuracy.


```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams['figure.figsize'] = (12, 7)
sns.set_style("whitegrid")

month_order = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
df['arrival_date_month'] = pd.Categorical(df['arrival_date_month'], categories=month_order, ordered=True)
plt.figure(figsize=(9,6))
ax = sns.countplot(data=df, x='is_canceled', palette=['#6A5ACD', '#FFD700'])
plt.title('Overall Booking Cancellation Rate', fontsize=20, fontweight='bold', pad=20)
plt.xticks([0,1], ['Not Canceled', 'Canceled'], fontsize=12)
plt.ylabel('Number of Bookings', fontsize=12)
plt.xlabel('')
total = len(df)
for i, p in enumerate(ax.patches):
    percentage = f'{100 * p.get_height() / total:.1f}%'
    ax.text(p.get_x() + p.get_width()/2., p.get_height() + 1000, percentage,
            ha='center', fontsize=16, fontweight='bold', color='#4B0082')
plt.ylim(0, max([p.get_height() for p in ax.patches]) * 1.15)
plt.show()
```

    /tmp/ipython-input-3756100489.py:13: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      ax = sns.countplot(data=df, x='is_canceled', palette=['#6A5ACD', '#FFD700'])



    
![png](output_16_1.png)
    


# the Average Daily Rate (ADR) by City

Across all 15 Indian cities, the **median ADR is remarkably consistent** between ~₹3,500 and ~₹5,500, showing that regular room rates are quite uniform nationwide. **Goa** and **Mumbai** achieve the highest medians, followed closely by Delhi and Bangalore, confirming their premium positioning. However, there is a **massive outlier at ~₹52,000+** (visible as the lone diamond point above ₹50,000), very likely a luxury suite booking, ultra-luxury villa in Goa, or data-entry error (e.g., ₹5,200 mistakenly recorded as ₹52,000). This extreme outlier should be investigated and either corrected or excluded from regular ADR analysis, as it distorts city-level comparisons and revenue-per-available-room calculations. Excluding this outlier, the highest realistic rates still belong to Goa, Mumbai, and Delhi, making them the most profitable markets.


```python
plt.figure(figsize=(12,7))
sns.boxplot(data=df, x='city', y='adr', order=df.groupby('city')['adr'].median().sort_values(ascending=False).index)
plt.title('Average Daily Rate (ADR) by City', fontsize=18)
plt.xticks(rotation=45)
plt.ylabel('ADR (₹)')
plt.xlabel('Cities')
plt.tight_layout()
plt.show()
```


    
![png](output_18_0.png)
    


# Bookings & Cancellations by Month
October emerges as the clear peak season with the highest volume of bookings and confirmed stays, closely followed by November and December, making it the prime window for dynamic pricing and strict non-refundable policies, while January and February stand out as the most reliable low-cancellation months ideal for corporate and long-stay contracts. Despite strong demand in the April–June summer vacation period, absolute cancellations are highest during these months as travelers book early and later change plans, representing the single biggest revenue-protection opportunity. Overall, the cancellation rate remains remarkably stable at ~30–33% across the entire year, meaning seasonality drives volume far more than customer behavior—hotels should therefore maintain aggressive overbooking and deposit strategies year-round rather than easing up in the off-season.


```python
plt.figure()
sns.countplot(data=df, x='arrival_date_month', hue='is_canceled',
        palette=['#A7D8F0', '#E4C1F9']

)
plt.title('Bookings & Cancellations by Month', fontsize=20, fontweight='bold', color='#008080')
plt.xticks(rotation=45)
plt.legend(['Not Canceled', 'Canceled'], title='Status', fontsize=11)
plt.ylabel('Number of Bookings')
plt.show()
```


    
![png](output_20_0.png)
    



# The lead time distribution (Days)
the overwhelming majority of bookings are made within 0–30 days of arrival, with cancellations (purple) heavily concentrated in this short-lead window — the longer guests book in advance, the higher the cancellation probability becomes. Beyond ~75 days, almost all bookings that survive to 200+ days end up canceled (orange almost disappears), making ultra-long lead times a strong red flag for revenue risk. Hotels can confidently treat bookings with lead time > 90 days as high-risk and apply stricter deposit rules or higher rates, while last-minute bookings (< 30 days) are the most reliable and should be actively encouraged through dynamic pricing and promotions. This single chart justifies a tiered cancellation policy based purely on lead time.


```python
df_lead = df[df['lead_time'] < 400]
plt.figure()
sns.histplot(data=df_lead, x='lead_time', hue='is_canceled', bins=50,
             palette=['#9B59B6', '#E74C3C'], alpha=0.8, edgecolor='white', linewidth=0.5)
plt.title('Lead Time Distribution (Days)', fontsize=20, fontweight='bold', color='#8E44AD')
plt.xlabel('Lead Time (days)')
plt.legend(['Not Canceled', 'Canceled'])
plt.show()
```


    
![png](output_22_0.png)
    


# Cancellation Rate by Market Segment & Customer Type
Transient and Transient-Party customers booked via Online TA or Groups show extreme cancellation rates (38–100%), while Contracts, Direct, Corporate and Complementary segments are rock-solid (<20%). The deadly combo is Groups + Transient (95%+ cancellation) and Undefined + Transient-Party (100%). Focus revenue protection almost entirely on Online TA and Group channels — everything else is safe.


```python
pivot = df.pivot_table(values='is_canceled', index='market_segment',
                       columns='customer_type', aggfunc='mean') * 100

plt.figure(figsize=(10,6))
sns.heatmap(pivot, annot=True, fmt='.1f', cmap="Blues", linewidths=1, linecolor='white',
            cbar_kws={'label': 'Cancellation Rate (%)'})
plt.title('Cancellation Rate by Market Segment & Customer Type', fontsize=18, fontweight='bold', color='#003366')
plt.show()
```


    
![png](output_24_0.png)
    


# New vs Repeated Guests – Cancellation Behavior
Repeated guests are pure gold: they represent <5% of bookings yet cancel <3% of the time, while new guests drive virtually all cancellations (~32% rate). Prioritize loyalty programs, personalized offers, and direct booking perks for repeaters — they are 10x+ more reliable and should be treated as your most valuable segment.


```python
plt.figure()
sns.countplot(
    data=df,
    x='is_repeated_guest',
    hue='is_canceled',
    palette=['#A3D5FF', '#FFB3C1']
)
plt.title('New vs Repeated Guests – Cancellation Behavior', fontsize=20, fontweight='bold', color='#2C3E50')
plt.xticks([0,1], ['New Guest', 'Repeated Guest'], fontsize=12)
plt.legend(['Not Canceled', 'Canceled'])
plt.show()

```


    
![png](output_26_0.png)
    


# Total Bookings Over Time
Total monthly bookings show a clear downward trend throughout 2024, dropping ~15–20% from ~10,000 in January to ~8,000 by year-end, with no seasonal peaks — indicating weakening demand or market saturation. This steady decline is a red flag: hotels must urgently cut costs, boost direct marketing, launch aggressive promotions, or risk severe under-occupancy in 2025.


```python

df['arrival_date'] = pd.to_datetime(
    df['arrival_date_year'].astype(str) + '-' +
    df['arrival_date_month'].astype(str) + '-' +
    df['arrival_date_day_of_month'].astype(str),
    format='%Y-%B-%d',
    errors='coerce'
)

print("Invalid dates:", df['arrival_date'].isna().sum())  #

monthly = df.resample('M', on='arrival_date').size()

plt.figure(figsize=(14,7))
monthly.plot(color='#9B59B6', linewidth=3.5)
plt.fill_between(monthly.index, monthly, color='#9B59B6', alpha=0.3)
plt.title('Total Bookings Over Time', fontsize=22, fontweight='bold', color='#8E44AD')
plt.ylabel('Number of Bookings', fontsize=13)
plt.xlabel('Date', fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

    Invalid dates: 0


    /tmp/ipython-input-3979548963.py:11: FutureWarning: 'M' is deprecated and will be removed in a future version, please use 'ME' instead.
      monthly = df.resample('M', on='arrival_date').size()



    
![png](output_28_2.png)
    


# Conclusion
This Indian hotel dataset reveals a challenging but actionable reality: a consistently high 31–32% cancellation rate, declining booking volume throughout 2024, and a heavy reliance on risky channels (Online TA, Groups), which drive massive revenue leakage. However, clear paths to recovery exist — repeated guests rarely cancel, short-lead and direct bookings are highly reliable, and peak months (Oct–Dec) offer strong demand. Immediate priorities: (1) shift volume to direct & repeated guests via loyalty programs and perks, (2) enforce non-refundable rates and deposits on high-risk segments (long lead time, Online TA, Groups), (3) aggressively overbook by 25–30%, and (4) launch urgent demand-stimulation campaigns to reverse the 2024 decline. Act on these insights now, and you can turn a 32% loss into protected, predictable revenue.


```python

```
