import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

pipe = pickle.load(open('pipe21.pkl', 'rb'))
recommender_pipe = pickle.load(open('recommender_pipeline (3)1.pkl', 'rb'))
df_dict = pickle.load(open('new_df_dict1.pkl', 'rb'))
df = pd.DataFrame(df_dict)
df = df.drop('tags', axis=1)
vectors = pickle.load(open('vectors1.pkl', 'rb'))
st.title('Gurgaon flat/house price prediction')
property_type = ['flat', 'house']
# flat->0
# house->1
sector_list = sorted(['sector 3', 'sector 35', 'sector 61', 'sector 92', 'sector 36',
                      'sector 26', 'sector 104', 'sector 88', 'sector 78', 'sector 81',
                      'sector 79', 'sector 33', 'sector 90', 'sector 108', 'sector 62',
                      'sector 102', 'sector 89', 'sector 113', 'sector 65', 'sector 48',
                      'sector 37', 'sector-3', 'sector 110', 'sector 43', 'sector-68',
                      'sector 109', 'sector 11', 'sector 2', 'sector 106', 'sector 82',
                      'sector 85', 'sector 10', 'sector 1', 'sector 22', 'sector 84',
                      'sector 71', 'sector-7', 'sector 77', 'sector 67', 'sector 57',
                      'sector 95', 'sector 99', 'sector 103', 'sector 4', 'sector 49',
                      'sector 30', 'sector 66', 'sector 15', 'sector 5', 'sector 13',
                      'sector 93', 'sector 68', 'sector-33', 'sector 63', 'sector-88',
                      'sector 86', 'sector 6', 'sector 52', 'sector 107', 'sector 12',
                      'sector 9', 'sector 25', 'sector 91', 'sector 70', 'sector 7',
                      'sector 47', 'sector 54', 'sector 111', 'sector 28', 'sector 41',
                      'sector - 89', 'sector 73', 'sector 56', 'sector 83', 'sector 53',
                      'sector 72', 'sector 69', 'sector-14', 'sector 50', 'sector-61',
                      'sector 24', 'sector 59', 'sector 38', 'sector 112', 'sector 45',
                      'sector 74', 'sector-82', 'sector 60', 'sector 51', 'sector-90',
                      'sector-81', 'sector-109', 'sector-108', 'sector 39', 'sector 58',
                      'sector 27', 'sector 80', 'sector 55', 'sector-92', 'sector-104',
                      'sector-107', 'sector-79', 'sector 23', 'sector 76', 'sector 31',
                      'sector 32', 'sector 21', 'sector-2', 'sector-5', 'sector-84',
                      'sector 105', 'sector 14', 'sector-69', 'sector 40', 'sector 29',
                      'sector 17', 'sector 101', 'sector 8', 'sector-57', 'sector-105',
                      'sector 42', 'sector -72', 'sector 34', 'sector 94', 'sector 46'])
bed_rooms = sorted([2, 3, 4, 1, 5, 6, 10, 9, 8, 7])
bath_rooms = sorted([2, 3, 4, 5, 1, 6, 7, 10, 9, 8, 11, 12])
balconies = ['1', '3', '2', '3+', 'No']
age_type = ['Relatively New', 'Old Property', 'New Property', 'Under Construction', 'Moderately Old']
servant_room = ['Yes', 'No']
store_room = ['Yes', 'No']
furnishing_type = ['Unfurnished', 'Semi Furnished', 'Full_Furnished']
luxury_category = ['Low', 'Medium', 'High']
floor_category = ['Mid Floor', 'Low Floor', 'High Floor']

col1, col2, col3 = st.columns(3)
with col1:
    property_type_selected = st.selectbox('Select Property Type', property_type)
    sector_selected = st.selectbox('Select Sector', sector_list)
    age_selected = st.selectbox('Select Property Status', age_type)
with col2:
    selected_bedroom_num = st.selectbox('Select Number of Bedrooms', bed_rooms)
    selected_bathroom_num = st.selectbox('Select Number of Bathrooms', bath_rooms)
    selected_balconies_num = st.selectbox('Select Number of Balconies', balconies)
with col3:
    furnish_type_selected = st.selectbox('Select Furnish Type', furnishing_type)
    if furnish_type_selected == 'Unfurnished':
        furnish_type_selected_tr = 0
    elif furnish_type_selected == 'Semi Furnished':
        furnish_type_selected_tr = 1
    else:
        furnish_type_selected_tr = 2

    luxury_type_selected = st.selectbox('Select Luxury Category', luxury_category)

    floor_type_selected = st.selectbox('Select Floor Category', floor_category)
col4, col5, col6 = st.columns(3)
with col4:
    sel_servant_room = st.selectbox('Want Servant Room', servant_room)
    if sel_servant_room == 'Yes':
        sel_servant_room_tr = 1
    else:
        sel_servant_room_tr = 0

with col5:
    sel_store_room = st.selectbox('Want Store Room', store_room)
    if sel_store_room == 'Yes':
        sel_store_room_tr = 1
    else:
        sel_store_room_tr = 0
with col6:
    area = int(st.number_input('Enter Built Up area in Sq.ft'))

input_df = pd.DataFrame({'property_type': [property_type_selected], 'sector': [sector_selected],
                         'bedrooms': [selected_bedroom_num], 'bathrooms': [selected_bathroom_num],
                         'balconies': [selected_balconies_num], 'agePossession': [age_selected],
                         'built_up_area': [area], 'servant_room': [sel_servant_room_tr],
                         'store_room': [sel_store_room_tr], 'furnishing_type': [furnish_type_selected_tr],

                         'luxury_category': [luxury_type_selected], 'floor_category': [floor_type_selected]})
input_recommended = pd.DataFrame({'property_type': [property_type_selected], 'sector': [sector_selected],
                                  'bedrooms': [str(selected_bedroom_num)], 'bathrooms': [str(selected_bathroom_num)],
                                  'balconies': [selected_balconies_num], 'agePossession': [age_selected],
                                  'built_up_area': [str(area)], 'servant_room': [str(sel_servant_room_tr)],
                                  'store_room': [str(sel_store_room_tr)],
                                  'furnishing_type': [str(furnish_type_selected_tr)],

                                  'luxury_category': [luxury_type_selected], 'floor_category': [floor_type_selected]})


def convert(data_frame):
    data_frame['agePossession'] = data_frame['agePossession'].apply(lambda x: x.lower())
    data_frame['luxury_category'] = data_frame['luxury_category'].apply(lambda x: x.lower())
    data_frame['floor_category'] = data_frame['floor_category'].apply(lambda x: x.lower())
    data_frame['agePossession'] = data_frame['agePossession'].apply(lambda x: x.replace(" ", ""))
    data_frame['floor_category'] = data_frame['floor_category'].apply(lambda x: x.replace(" ", ""))
    data_frame['sector'] = data_frame['sector'].apply(lambda x: x.replace(" ", "").replace("-", ""))
    l = [row for row in data_frame.iloc[0]]
    return l


my_string = convert(input_recommended)
my_string = " ".join(my_string)
vectorized_string = recommender_pipe.transform([my_string])


def recommended(test_home):
    empty = []
    for i in sorted(list(enumerate(cosine_similarity([test_home], vectors)[0])), reverse=True, key=lambda x: x[1])[1:6]:
        empty.append(i[0])
    return df.iloc[empty]


result_for_recommendation = recommended(vectorized_string.toarray()[0])
btn1 = st.button('Predict Price')

if btn1:
    if area > 0:
        result = pipe.predict(input_df)
        z = round(np.expm1(result[0]), 4)
        if z >= 1:
            st.write('Predicted Price is :', round(z, 2), 'crores')
        elif z < 1:
            st.write('Predicted Price is :', round(z * 100,2), 'lakhs')

        st.title('Recommended Properties')
        col6, col7, col8 = st.columns(3)
        with col6:
            st.write('Society')
        with col7:
            st.write('Sector')
        with col8:
            st.write('Selling Price')

        for index, row in result_for_recommendation.iterrows():

            with col6:
                st.write(row['society'].title())
                st.write(" ")
            with col7:
                text = row['sector']
                # Check if the text starts with "sector" (case-insensitive)
                if text.lower().startswith("sector"):
                    # Capitalize the "sector" part and concatenate it with the rest of the text
                    text = "Sector" + " " + text[len("sector"):]

                st.write(text)

                st.write(" ")
            with col8:
                st.write(row['price_in_cr'], 'Crore')
                st.write(" ")
    else:
        st.write("Invalid Built Up area")
