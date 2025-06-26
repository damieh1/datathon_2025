
# Challenge #1: Create a Discourse Dataset from X/Twitter</strong></summary>

---

Welcome to **Challenge #1** of the 2025 Datathon: **Creating a Discourse Dataset from X (formerly Twitter)**

In this task, your goal is to collect, clean, and annotate a small but focused dataset of tweets related to antisemitism and Israel-related discourse. You’ll gain practical experience with platform search, data scraping, and structured annotation. This ReadMe guides you through collecting X/Twitter tweet data using Bright Data and preparing it for upload to our annotation portal: [annotate.osome.iu.edu](https://annotate.osome.iu.edu).


---

## What You'll Be Doing
- Use **X/Twitter's Advanced Search** to identify relevant tweets
- Collect tweet URLs based on defined timeframes
- Scrape post content using **Bright Data**
- Clean and filter the dataset based on your keyword(s)
- Upload and annotate the dataset using the **[Annotation Portal](https://annotate.osome.iu.edu)**

---

## Prerequisites

To complete this challenge, make sure you have:

- An active **X (Twitter)** account  
- Familiarity with **X’s Advanced Search**  
  *(https://twitter.com/search-advanced)*  
- A **Google account** for using **Google Colab**  

---

## How to Generate a Discourse Dataset from X (formerly Twitter)

1. [Search for a Specific Keyword or Hashtag](#1-search-for-a-specific-keyword-or-hashtag)  
2. [Collect Data from a Broad Variety of Different Users](#2-collect-data-from-a-broad-variety-of-different-users)  
3. [Format for Bright Data Scraping](#3-format-for-bright-data-scraping)  
4. [Clean and Parse Your Scraped Tweet Data](#4-clean-and-parse-your-scraped-tweet-data)  
5. [What to Submit](#5-what-to-submit)
6. [Working with Colab](#6-working-with-data-output-in-google-colab)
7. [Further Notes](#7-further-notes)


---

## Objective

Your goal is to generate a dataset that sheds light on the online discourse surrounding a **specific keyword or hashtag** (e.g., "Jews", "Israel", or "Zionists") by collecting posts from **X’s “Top” and “Latest” tweets** across **a defined time period**.  

You will then submit the list of tweet URLs to **Bright Data** for structured scraping.

---

## Step-by-Step Instructions

## 1. Search for a Specific Keyword or Hashtag


We recommend starting with the [**Advanced Search**](https://twitter.com/search-advanced) feature on **X.com** and using **newly created accounts** to avoid personalized search bias.

- **Search for at least one or multiple keywords** (e.g., `Jews`, `#Zionists`, etc.)
- Use the **“Top” tab** in X’s search results for high-engagement posts
- Adjust the filters to a specific time range (see below)
- Open the user profile in new tabs and copy the full **accound URL**

> Example Account URL:  
> `https://x.com/elonmusk/`

---

## 2. Collect Data from a Broad Variety of Different Users

To ensure your dataset includes a wide range of perspectives and accounts, we recommend searching across multiple timeframes when collecting tweet URLs.

Even though you cannot apply strict date filters during scraping, using **X/Twitter’s Advanced Search** with different date ranges during manual collection can help you:

- Capture more diverse voices
- Avoid over-representation from a single trending event
- Ensure better distribution of accounts in your dataset

**Important:** Your goal is to collect **100 tweets total**; consider dividing your search into **weekly or 2–3 day intervals**.

| Example Time Ranges (September 2024) |                       |
|--------------------------------------|-----------------------|
| Week 1                               | September 1–7, 2024   |
| Week 2                               | September 8–13, 2024  |
| Week 3                               | September 14–21, 2024 |
| Week 4                               | September 22–29, 2024 |

#### Guidelines:
- Collect **100 tweets** total that include the **exact keyword or hashtag** you selected
- Use **X’s Advanced Search** to vary the date range during collection
- Aim for a wide variety of **different user accounts**
- Prioritize tweets with **200+ views**, but lower-view tweets can be included when relevant
  
---

## 3. Format for Bright Data Scraping

Once you’ve collected all tweet URLs (in a `.csv` or `.txt` file), prepare to scrape full post content using **Bright Data**.

1. Go to:  
   `Bright Data > Web Scrapes > X (Twitter) > Posts > Discover by URL`

2. Click:  
   **Add Inputs** → **Upload CSV** → **Clear All**

3. Required input format:

| URL                              | max_number_of_posts            |
|----------------------------------|--------------------------------|
| Full Account URL (one per row)   |Use a fixed value like `100`    |

> Example CSV structure:
>
> ```csv
> url,max_number_of_posts
> https://x.com/username1,100
> https://x.com/someuser2,100
> ```

> To reach the desired frequency of the dataset including the keyword or hashtag, you may need to scrape multiple hundreds of tweets. Keep in mind that this process is iterative and time-consuming and may require you to expand your list of accounts further.*

---

## 4. Clean and Parse Your Scraped Tweet Data

After scraping content from user profiles with Bright Data, your output will contain many metadata fields about the account and the scraped posts. One important field is:

- `posts`: This contains a list of tweets returned for each profile as a **JSON string**.

Your task is to extract individual tweets from this `posts` column and isolate the tweets that actually contain the keyword or hashtag you selected.



#### Example Output Columns from Bright Data

Here’s a partial list of the fields you will find in the Bright Data output:

##### **Profile-Level Fields**
- `x_id`: internal user ID  
- `url`: profile URL  
- `id`: Twitter handle  
- `profile_name`, `biography`, `is_verified`  
- `profile_image_link`, `external_link`, `date_joined`  
- `followers`, `following`, `subscriptions`, `posts_count`  
- `is_business_account`, `is_government_account`  
- `location`, `birth_date`, `category_name`  
- `timestamp`: scrape timestamp  
- `input`: original scrape parameters

##### **Post-Level (Nested in `posts`)**
Each tweet object inside `posts` includes:
- `post_id`: **tweet ID**  
- `description`: **tweet text**  
- `post_url`: full URL to the tweet  
- `date_posted`  
- `likes`, `reposts`, `replies`, `views`  
- `photos`: list of media image URLs  
- `videos`: video info (if any)  
- `hashtags`: extracted hashtags (if available)

## Process Your Bright Data Output in Google Colab

### Uploading Your File in Colab

When you open your notebook in Google Colab, follow these steps to upload your Bright Data output file:

1. Click the **folder icon** (📁) on the left-hand side.
2. Open the folder named **`content/`**.
3. **Right-click** on the `content` folder — a dropdown menu will appear.
4. Click **“Upload”** and select your Bright Data `.csv` file(s).
5. Once uploaded, run the code snippet below to format your data.

#### Parsing Script: Extract Tweets and Format for Annotation

We recommend using **Python** 🐍 for this task.  

- Please see the code snippet below for parsing and extracting the Tweet ID and description for each tweet.

*What the script does* — It processes the `.csv` file you receive from **Bright Data** and extracts:
  1. all relevant tweet-level information
  2. filters posts by your chosen keywords or hashtags
  3. saves as a `.csv` for further filtering and annotation.

Important: *Check [Section 6](#6-working-with-data-output-in-google-colab) to learn how to run the code without installing Python on your personal computer.*
```python
# Load Libraries
import pandas as pd
import json

# Load your Bright Data output file
df = pd.read_csv("/content/sample_data/brightdata.csv")

# Define your search terms (case-insensitive)
keywords = ["jew", "netanyahu", "#gaza", "Zionist", "Israel", "settlers"]  # 👈 Edit this list as needed

# Storage for filtered results
parsed_rows = []
text_id = 1

# Iterate through each row
for _, row in df.iterrows():
    try:
        username = row.get("id", "")
        posts_raw = row.get("posts", "")

        if not posts_raw or pd.isna(posts_raw):
            continue

        # Safely parse JSON from the 'posts' column
        try:
            posts = json.loads(posts_raw)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw content preview: {posts_raw[:300]}\n")
            continue

        for post in posts:
            description = post.get("description", "")
            tweet_id = post.get("post_id", "")
            date_posted = post.get("date_posted", "")

            if description and tweet_id:
                if any(kw.lower() in description.lower() for kw in keywords):
                    parsed_rows.append({
                        "text_id": text_id,
                        "Text": description,
                        "tweet_id": tweet_id,
                        "Username": username,
                        "date_posted": date_posted
                    })
                    text_id += 1

    except Exception as e:
        print(f"Unexpected error: {e}")

# Save the filtered output
df_out = pd.DataFrame(parsed_rows)
df_out.to_csv("/content/sample_data/parsed_tweets_filtered.csv", index=False, encoding="utf-8")

# Shows you how much Tweets your collection already contains
print(f"Parsing complete. {len(df_out)} relevant tweets saved.")


```

#### The Script will generate the following `.csv` output
| text_id | Text                                                        | tweet_id             | Username     | date_posted            |
|---------|-------------------------------------------------------------|----------------------|--------------|------------------------|
| 1       | Tweet text including specified keyword                      | 1545842080431321088  | username     | 2022-10-28T03:49:11Z   |
| 2       | Tweet text including specified keyword                      | 1519480461749016577  | username     | 2022-04-28T00:56:58Z   |
| 3       | Tweet text including specified keyword                      | 1212256198588662068  | username     | 2024-07-13T22:45:13Z   |
| 4       | Tweet text including specified keyword                      | 1586104694428859648  | username     | 2022-10-28T21:16:42Z   |
| 5       | Tweet text including specified keyword                      | 1854209929529247803  | username     | 2024-11-06T16:39:24Z   |

---

## 5. What to Submit

Each team should submit the following:

##### A. Cleaned Annotation Dataset (.csv)

*After preprossing --> Prepare the final dataset for upload to our annotation portal*

> ⚠️ **Important:**  
> You may need to process **multiple Bright Data files** until you reach a sample of 100 relevant tweets.  
> Always ensure your output includes **only tweets containing your selected keyword or hashtag** and follows the required format.

---

Once you've parsed your tweets, you can transform the output into the required format for the annotation portal.

### 🐍 Code: Convert to Annotation-Ready Format

- This code generates a `.csv` file with the two required columns: `TweetID` and `Username`.

```python
import pandas as pd

# Step 1: Load the uploaded file
# Adjust the path if necessary — this assumes you're using the default Colab folders
parsed_df = pd.read_csv("/content/sample_data/parsed_tweets_filtered.csv")

# Step 2: Transform into the required format
transformed_df = parsed_df.rename(columns={"tweet_id": "TweetID"})[["TweetID", "Username"]]

# Step 3: Save and download the result
output_file = "/content/sample_data/parsed_tweets_for_annotation.csv"
transformed_df.to_csv(output_file, index=False)
files.download(output_file)
```


- File must contain only tweets that include **one or more of your selected keywords or hashtags**
- Format the file as follows:

| TweetID               | Username     |
|------------------------|--------------|
| 1532051756166926336    | isca_iu      |
| 1176676441666347008    | OSoMe_IU     |
| 1808854301042761934    | IndianaUniv  |

- Column names must be **TweetID** and **Username** (case-sensitive)
- Save file as: `your_final_annotation_input.csv` (specify this with your keyword)
- This file is ready to be uploaded to our annotation portal at [annotate.osome.iu.edu](https://annotate.osome.iu.edu)



##### B. Report (PDF or Markdown)
- A brief report (1–2 pages) that includes:
  - The keyword(s) or hashtag(s) used
  - Your rationale for selecting them
  - The date range used and why
  - Notes on manual filtering or scraping inconsistencies
  - Total number of tweets after filtering
  - The amount of antisemitic or biased content within the dataset
  - The annotation scheme that was used as a standardized annotation guideline, e.g., based on the [IHRA-WDA](https://github.com/AnnotationPortal/DatathonandHackathon.github.io/blob/main/guides/annotation_scheme.md)


---

### Important Reminder

- Ensure that your **Tweet IDs are stored as strings**, not numbers, to avoid truncation or formatting issues in Excel or Google Sheets. When in doubt, open the file in a plain text editor or use `.csv` in UTF-8.

---

### 6. Working with Data Output in Google Colab

To parse, filter, and prepare your tweet dataset using the provided **Python** 🐍 script, we recommend using **Google Colab**. 

***Colab** allows you to run the parsing script in [Section 4](#4-clean-and-parse-your-scraped-tweet-data) without installing Python on your personal computer.*

#### Steps:
1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Sign in using a **Google account** (this is required to upload files)
3. Click **"New Notebook"**
4. Upload your Bright Data `.csv` file:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```
5. Copy & Paste the code snipped *from Section 4* into the Notebook and run the code
7. Share the script with your team if you want to collaborate
   
---

## 7. Further Notes

- You will work and collaborate in small teams up to 4—5 members
- Be consistent: only collect tweets containing the **exact keywords or hashtags**
- Avoid collecting identical content, e.g. retweets. Although retweets are generally not excluded, duplicates of this content type should not appear multiple times within the dataset.

####  How to View a Tweet Using Only Its ID

If you only have the `TweetID` (and not the full URL), you can still access the original tweet in your browser by using the following helper site:

Go to: [https://www.bram.us/](https://www.bram.us/2017/11/22/accessing-a-tweet-using-only-its-id-and-without-the-twitter-api/)

##### Instructions:
1. Copy any Tweet ID from your dataset (e.g., `1585811080431321088`)
2. Paste it into the helper site linked above
3. It will redirect you to the full tweet

---
<div style="font-size: 0.2em; color: #555; margin-top: 2em;">
  <p>
    All materials in this repository were created by <strong>Daniel Miehling</strong> (Indiana University / ISCA), unless otherwise noted.
  </p>
  <p>
    Please contact <a href="mailto:damieh@iu.edu">damieh@iu.edu</a> for reuse permissions or collaboration inquiries.
  </p>
</div>

