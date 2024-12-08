import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from datetime import datetime

st.set_page_config(page_title='Singapore Resale Flat Prices Prediction', 
                       page_icon=None, layout="wide", 
                       initial_sidebar_state="expanded", 
                       menu_items=None)
st.markdown('''
        <style>
        [data-testid="stHeader"] {
        height: 0;
        
        }
        div.block-container {
        padding: 2rem;
    }
    </style>''',unsafe_allow_html=True
    )
st.header('Singapore Resale Flat Prices Prediction',divider = 'grey')
st.markdown(
        """ <style>
            #singapore-resale-flat-prices-prediction {
            text-align: center;
            }
            </style>""",unsafe_allow_html=True
    )
def predict_datapoint(main_container):
    col1,col2,col3 = main_container.columns(3,gap="medium")
    with col1:
        town_list = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
                    'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
                    'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
                    'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
                    'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
                    'TOA PAYOH', 'WOODLANDS', 'YISHUN', 'LIM CHU KANG']
        container=st.container(border = True)
        town = container.selectbox("Select the town",town_list,index=None,placeholder="-- Select  --")
        container=st.container(border = True)
        floor_area_sqm = container.number_input("Enter the floor area in square meter",min_value=28,max_value=500,value="min"
                                       ,key="floor_area_sqm1",step=1)
        
        
        container=st.container(border = True)
        storey_start = container.number_input("Enter the starting storey",min_value=1,max_value=50,value="min"
                                       ,key="storey_start1",step=1)
    
    with col2:

        flat_type_list = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', '1 ROOM','MULTI GENERATION']
        container = st.container(border=True)
        flat_type = container.selectbox("Select the flat type",flat_type_list,index=None,placeholder="-- Select  --")
        container = st.container(border=True)
        remaining_lease =container.number_input("Enter the reamining lease years",min_value=1,max_value=99,value="min"
                                       ,key="remaining_lease1",step=1)
        container=st.container(border = True)
        storey_end = container.number_input("Enter the ending storey",min_value=1,max_value=50,value="min"
                                       ,key="storey_end1",step=1)
    with col3:
        flat_model_list = [ 'Improved', 'New Generation', 'Dbss', 'Standard', 'Apartment',
                            'Simplified', 'Model A', 'Premium Apartment', 'Adjoined Flat',
                            'Model A-Maisonette', 'Maisonette', 'Type S1', 'Type S2',
                            'Model A2', 'Terrace', 'Improved-Maisonette', 'Premium Maisonette',
                            'Multi Generation', 'Premium Apartment Loft', '2-Room', '3Gen']
        container=st.container(border = True)
        flat_model = container.selectbox("Select the flat model",flat_model_list,index=None,placeholder="-- Select  --")
        container = st.container(border=True)
        block = container.number_input("Enter the block",min_value=1,max_value=999,value="min"
                                       ,key="block1",step=1)
        container = st.container(border=True)
        current_year = datetime.now().year
        year =container.number_input("Enter the current year",min_value=1990,max_value=current_year,value=current_year
                                       ,key="year1",step=1)
    
    # col1,col2,col3 = st.columns(3,gap ='large')
    if town and flat_type and block and floor_area_sqm and flat_model and remaining_lease and year and storey_start and storey_end:
        # with col2:
            if st.button("CLICK  TO  PREDICT  THE PRICE",use_container_width=True):
                data = CustomData(town=town, flat_type=flat_type, flat_model=flat_model, block=block, floor_area_sqm=floor_area_sqm, remaining_lease=remaining_lease, year=year,storey_end=storey_end,storey_start=storey_start)
                pred_df = data.get_data_as_data_frame()
                predict_pipeline = PredictPipeline()
                results = predict_pipeline.predict(pred_df)
                st.markdown(
                            f"""
                            <div style="text-align: center;">
                                <h2>✨ Your Predicted Resale Price: 
                                <span style="color: skyblue; font-weight: bold;">$<strong>  {results[0]:,.2f} </strong></span>
                                </h2>
                                <hr style="border: 1px solid grey; width: 50%; margin: auto;">
                                <p style="font-size:18px;">🏡 Here's the estimated price for your flat based on the details you provided!</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                return results
        

if __name__ == '__main__':
    main_container = st.container(border = True)
    predict_datapoint(main_container)