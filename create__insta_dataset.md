# Task: Instagram Data Collection and Annotation

This task guides you through collecting Instagram post data using Bright Data and preparing it for upload to our annotation portal: [annotate.osome.iu.edu](https://annotate.osome.iu.edu).

---

## Step-by-Step Instructions

### 1. Find Suspicious Accounts

- Use the Instagram **web app** to search for relevant topics or hashtags such as:
  - `Jews`, `Israel`, `Zionists`, or hashtags like `#Palestine`, `#FreeGaza`, `#FromTheRiverToTheSea`
- As a team, select **at least 10 accounts** that post ideologically charged or suspicious content.
- Copy the full profile URLs (e.g., `https://www.instagram.com/exampleuser/`).

---

### 2. Scrape Data via Bright Data

Use Bright Data’s automated webscraping tools:

**Navigate to:**
> Web Scrapes → Instagram – Posts – Discover by URL

For each user, fill in the following fields:

| Field                  | Description                                                  |
|------------------------|--------------------------------------------------------------|
| `url`                  | Full Instagram profile URL                                   |
| `num_of_posts`         | Number of posts to scrape from this profile                  |
| `posts_to_not_include` | *(optional)* Leave blank unless excluding pinned posts       |
| `start_date`           | Start of your timeframe (e.g., `2023-10-08`)                 |
| `end_date`             | End of your timeframe                                        |
| `post_type`            | Set to `feed` or leave blank for standard posts              |

Your final dataset should include **100 posts total** across all accounts.

---

### 3. Prepare Data for Upload

Your Bright Data output needs to be **transformed** into the following format for annotation:

| Text          | text_id |
|---------------|---------|
| Example text 1| 1       |
| Example text 2| 2       |

Each post's **caption** (or full text) should be stored under the `Text` column.  
Each post should have a unique `text_id` (start with 1 and increment).

Save your result as a `.csv` file using **UTF-8 encoding**.

---

### 4. Upload to Annotation Portal

Go to: [annotate.osome.iu.edu](https://annotate.osome.iu.edu)  
- Upload your `.csv` file
- Start annotating based on the project instructions
- Instructions on how to manage your annotation project will be provided during the first workshop on July 13

---

## What to Submit

-  `your_data.csv` (in the format shown above)


A short report with:
- Search strategy and keywords used
- Why you selected these accounts
- Your chosen timeframe and rationale
- Number of posts per account

---
