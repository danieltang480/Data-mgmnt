The dataset can be found here on Kaggle: https://www.kaggle.com/datasets/ravindrasinghrana/carbon-co2-emissions/data

The purpose for this project is mainly to inform others about the dangers of climate change. This project obviously does not aim to solve it, but I want to bring more awareness to the cause. 


Instructions: I created the database in the python script, with 
create_table_query = """
CREATE TABLE emissions_data (
    Country VARCHAR(255),
    Region VARCHAR(255),
    Date DATE,
    `Kilotons of Co2` FLOAT,
    `Metric Tons Per Capita` FLOAT
);
"""
This created the database, and then I filled the database with the data from the dataset. 
After running the python script, it would then tell me my intercepts, which were future predictions for the future and the climate, and also the mean squared coefficient, which would tell me errors.
