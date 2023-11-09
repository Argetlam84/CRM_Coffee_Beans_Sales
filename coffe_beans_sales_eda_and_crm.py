# Warning!!! Read the Conclusion below 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

# https://www.kaggle.com/datasets/saadharoon27/coffee-bean-sales-raw-dataset?rvi=1 ==> dataset url

df_o = pd.read_excel("/kaggle/input/coffee-bean-sales-raw-dataset/Raw Data.xlsx", sheet_name="orders")
df_c = pd.read_excel("/kaggle/input/coffee-bean-sales-raw-dataset/Raw Data.xlsx", sheet_name="customers")
df_p = pd.read_excel("/kaggle/input/coffee-bean-sales-raw-dataset/Raw Data.xlsx", sheet_name="products")
df_o.head() # order_table
df_c.head() # customer_table
df_p.head() # products_table

# Data Preparation

# dropping NAN variables if we not they will be a problem
df_o.head()
drop_nan_var = df_o[["Customer Name","Email","Country","Coffee Type","Size","Unit Price","Sales","Roast Type"]]
df_o.drop(drop_nan_var, axis=1, inplace=True)
df_o.head()

# merging 2 datasets on Customer ID
df_c.head()
df_co = df_o.merge(df_c, on="Customer ID")
df_co.head()

# merging 2 dataset for one dataset
df_p.head()
df = df_co.merge(df_p, on="Product ID")
df.head()

# Let's Get to Know the Data

# checking variables types
df.dtypes

# gathering together categorical columns
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["object","category","bool"]]
cat_cols

# gathering together numerical columns
num_cols = [col for col in df.columns if df[col].nunique() > 10 and df[col].dtypes not in ["object","category","bool"]]
num_cols

# detecting to seems like numeric but actually categorical variables
num_but_cat =  [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int","float"]]
num_but_cat

# detecting to seems like categorical but actually cardinal variables
cat_but_car = [col for col in df.columns if df[col].nunique() > 30 and str(df[col].dtypes) in ["object","category","bool"]]
cat_but_car

# gathering together categorical variables that we detect as categorical variables
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]
cat_cols

# Data Visualizations

# categorical variables ratio for the whole dataset and data visiluation with countplot
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("############################################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.xticks(rotation=45)
        plt.show(block=True)


cat_summary(df,"Country",plot=True)

cat_summary(df,"Coffee Type",plot=True)

cat_summary(df,"Quantity",plot=True)

# detecting to some quantiles and visiualation with histogram
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.xticks(rotation=45)
        plt.show(block=True)



num_summary(df,"Unit Price",plot=True)

num_summary(df,"Profit",plot=True)

df["Profit Rate"] = df["Profit"] / df["Unit Price"]
df["Profit Rate"].head(10)

df.groupby(["Size","Coffee Type"])["Profit Rate"].mean()

num_summary(df,"Profit Rate", plot=True)

# RFM Segmentation

# we need to monetary value for rfm metrics so we multiple unit price with quantity
df["Total Price"] = df["Quantity"] * df["Unit Price"]
df["Total Price"].head()

df["Order Date"].max()

# we are setting date for analysis
analysis_date = dt.datetime(2022,8,21)
analysis_date

# creating rfm variable
rfm = df.groupby("Customer ID").agg({"Order Date": lambda date: (analysis_date - date.max()).days,
                                    "Order ID": lambda num: num.nunique(),
                                    "Total Price": lambda total: total.sum()})
rfm.head()

# changing columns name as we know rfm metrics
rfm.columns = ["recency","frequency","monetary"]
rfm.columns

# creating recency and frequency scores
# we will not use monetary value so we did not include the monetary value
rfm["Recency Score"] = pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1])
rfm["Frequency Score"] = pd.qcut(rfm["frequency"].rank(method="first"),5,labels=[1,2,3,4,5])
rfm.head()

# gathering together recency and frequency scores as rf score
rfm["RF Score"] = rfm[["Recency Score", "Frequency Score"]].agg(lambda x: ''.join(x.astype(str)), axis=1)
rfm.head()

seg_map ={
    r"[1-2][1-2]":"hibernating",
    r"[1-2][3-4]":"at_risk",
    r"[1-2]5":"cant_loose",
    r"3[1-2]":"about_to_sleep",
    r"33":"need_attention",
    r"[3-4][4-5]":"loyal_customers",
    r"41":"promising",
    r"51":"new_customers",
    r"[4-5][2-3]":"potential_loyalists",
    r"5[4-5]":"champions"
}

# creating segments according to the rf scores
rfm["Segment"] = rfm["RF Score"].replace(seg_map,regex=True)
rfm.head()


rfm[rfm["Segment"] == "champions"]

rfm[rfm["Segment"] == "new_customers"]

cat_summary(rfm,"Segment",plot=True)

rfm.loc[rfm["Segment"] == "champions"]["frequency"].mean()

rfm.loc[rfm["Segment"] == "new_customers"]["frequency"].mean()

rfm.loc[rfm["Segment"] == "loyal_customers"]["frequency"].mean()

rfm.loc[rfm["Segment"] == "potential_loyalists"]["frequency"].mean()


# Conclusion
# The Coffee Beans Sales dataset was curated for CRM analytics and underwent comprehensive analysis.
# The conclusion drawn from the process revealed that the dataset's suitability for effective CRM analytics was limited.
# The primary issue identified was the lack of significant variation within the frequency variable.
# This limitation impeded the accurate execution of RFM (Recency, Frequency, Monetary) segmentation and classification methods.
# Consequently, the distinct differentiation of customer segments proved challenging. 





