# The assignment

The input for this assignment is an archive ( download **[here](https://drive.google.com/file/d/1jF7lnMUffCX8U252MoY7jowb7VedFOp8/view?usp=sharing)** ) containing 3 datasets with data about the same companies from 3 different sources: 

1. Facebook (facebook dataset.csv)
2. Google (google dataset.csv)
3. Company Website (website dataset.csv)

The final purpose of this exercise is to create a 4th dataset that contains the other 3 and, by joining them, we should reach a better accuracy on common columns. The columns that interest us the most are Category, Address(country, region...), Phone, Company names.


# Solution

I used PySpark to analyze and clean the datasets.

Read the data into pyspark dataframes and select columns that we need from each.

```python
fb_df = spark.read.csv("data/facebook_dataset.csv", header=True)
gg_df = spark.read.csv("data/google_dataset.csv", header=True)
wb_df = spark.read.csv("data/website_dataset.csv", header=True, sep=";")

# We don't need cols [description, email, link, page_type, phone_country_code]
fb_df = fb_df.select(
    "domain",
    "address",
    "categories",
    "city",
    "country_code",
    "country_name",
    "name",
    "phone",
    "region_code",
    "region_name",
    "zip_code",
)

# We don't need cols [phone_country_code, text, raw_phone]
gg_df = gg_df.select(
    "domain",
    "address",
    "category",
    "city",
    "country_code",
    "country_name",
    "name",
    "phone",
    "region_code",
    "region_name",
    "zip_code",
    "raw_address",
)

# We don't need cols [domain_suffix, language, tld]
# And I rename cols to be unifies with another datasets
wb_df = wb_df.select(
    F.col("root_domain").alias("domain"),
    F.col("s_category").alias("category"),
    F.col("main_city").alias("city"),
    F.col("main_country").alias("country_name"),
    F.col("legal_name").alias("name"),
    F.col("main_region").alias("region_name"),
    F.col("phone"),
    F.col("site_name"),
)
```

# Data Cleaning

First and foremost, I split all categories by the symbols "|", "&", "," and perform data denormalization so that each row contains information with a single category. This is because there are instances where, in Facebook, a company is listed under four categories, while in Google, it's listed under just one. To merge this data, denormalization is necessary.

Additionally, in each dataset, I clean the columns with phone numbers. Remove all characters, retaining only the digits.

For all columns, I convert to lowercase and trim any potential spaces at the beginning and end.

As a result, the dataset sizes have increased:

Facebook 72077 -> 145232
Google 356520 -> 580891
Web 72018 -> 111720

But then I also removed all rows where the 'name' column value is null:

Facebook 145232 -> 145216
Google 580891 -> 580843
Web 111720 -> 50602

And saved the results in parquet format.

Due to the change in format, data that was 153MB in CSV is now saved as 66MB in parquet.


# Functions for cleaning.


```python
from pyspark.sql import functions as F
from pyspark.sql import DataFrame

def split_by_symbol(df: DataFrame, sym: str, column: str) -> DataFrame:
    """
    Splits the values in the specified column of a DataFrame by a given symbol and explodes them into separate rows.
    
    :param df: Input DataFrame
    :param sym: Symbol to split by
    :param column: Column in which to look for the symbol
    :return: DataFrame with values in the specified column split by the symbol
    """
    df_no = df.filter(~F.col(column).contains(sym) | F.col(column).isNull())
    df_yes = df.filter(F.col(column).contains(sym))
    res = df_yes.withColumn(
        column, 
        F.explode(
            F.split(df_yes[column], sym)
        )
    )
    return res.distinct().unionByName(df_no)

def lower_and_trim(df: DataFrame) -> DataFrame:
    """
    Converts all the column values of a DataFrame to lowercase and trims any spaces.
    
    :param df: Input DataFrame
    :return: DataFrame with all values in lowercase and trimmed
    """
    return df.select([F.trim(F.lower(F.col(column))).alias(column) for column in df.columns])

def add_prefix_to_cols(df: DataFrame, prefix: str) -> DataFrame:
    """
    Adds a prefix to all the column names of a DataFrame.
    
    :param df: Input DataFrame
    :param prefix: Prefix string to be added
    :return: DataFrame with the prefixed column names
    """
    return df.select([F.col(column).alias(f"{prefix}_{column}") for column in df.columns])

def clean_phone(df: DataFrame, column: str) -> DataFrame:
    """
    Cleans phone number data by removing all non-numeric characters from a specified column in a DataFrame.
    
    :param df: Input DataFrame
    :param column: Column containing phone numbers to be cleaned
    :return: DataFrame with cleaned phone numbers
    """
    return df.withColumn(column, F.regexp_replace(column, "[^0-9]", ""))

```


# Search similar companies

Using this as a reference https://periodic-allspice-d61.notion.site/Soleadify-Project-818607ae78ba43819393c05ac7cdbb65

I decided to improve the search for similar company names. Instead of the Levenshtein algorithm, I used ML model that constructs a 768-dimensional vector and is capable of searching for similar sentences not by characters, but by the meanings described in the sentence.

https://huggingface.co/tasks/sentence-similarity




# The decisions I have made in order to join the 3 datasets include:

## 1. What column will you use to join?


## 2. If you have data conflicts once you join, which one do you believe?


## 3. If you have very similar data, what information will you keep?


