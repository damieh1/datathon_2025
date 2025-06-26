
# Datathon and Machine Learning Competition on Antisemitism

The Datathon and Machine Learning Competition will take place in **July 2025**. General information can be found here:  
https://isca.indiana.edu/publication-research/social-media-project/datathon-2025/index.html

## Table of Contents

1. [Challenge Overview](#challenge-overview)  
2. [Deliverables](#deliverables)  
3. [Prizes & Certificates](#prizes--certificates)  
4. [Resources and Past Events](#resources-and-past-events)

---


## Challenge Overview

- Before beginning your tasks, please read the full challenge description thoroughly: [2025 Task Instructions](https://github.com/damieh1/datathon_2025/blob/main/Datathon_Challenge.pdf)
- Also, check all the links in this ReadMe file.


### #1 Challenge: Dataset Creation (July 13–20, 2025)

More information will be provided at the workshop on July 20.

Goal: Create a small but meaningful labeled dataset for hate speech detection.

Tasks:
1. Use the [Bright Data](https://brightdata.com/products/web-scraper/functions) interface to scrape at least 100 relevant user-generated posts from [X.com](https://X.com).
2. Define your scraping focus (hashtags, user groups, topics) and document your rationale and potential biases.
3. Annotate your data using a structured definition of antisemitism and hate speech.
   - We provide a standardized annotation scheme based on the **IHRA Working Definition (IHRA-WDA)**, including paragraph-level categories such as conspiracy narratives, Holocaust distortion, and double standards toward Israel.
   - This scheme is available here: [Annotation Scheme (GitHub)]([https://github.com/damieh1/datathon_2025/blob/main/annotation_scheme.md)
   - Participants are expected to use this scheme or adapt it with justification.
   - However, if your team prefers to engage with alternative frameworks such as the [**Nexus Document**](https://nexusproject.us/nexus-resources/the-nexus-document/) or the [**Jerusalem Declaration on Antisemitism (JDA)**](https://jerusalemdeclaration.org), please note that these do not currently offer equivalent category-level structures. Therefore, any adaptation should include:
      - A mapping of concepts from the chosen framework into operational categories
      - A clear rationale for the adaptation
      - An explanation of how it supports reliable annotation

   All annotation efforts should go beyond binary classification and aim to capture the diversity and nuance of antisemitic expression in online content.
   
   You may complete the annotation using the [Annotation Portal](https://annotate.osome.iu.edu), which is specifically optimized for this task. A key advantage of the portal is that it retrieves tweets directly from X/Twitter and displays them in their original context, preserving layout, embedded media, and user metadata. This allows for more accurate annotation, especially when dealing with coded language or ambiguous references.

   - The functionality of the portal will be explained at the first workshop. You do not need to register in advance. Once you register, please make sure to remember your password.

   **IMPORTANT:** Please make sure you register to the portal with the same email address you used to register for this challenge. Please use "Datathon" for "Annotation Type" when registering. You can upload & annotate your scraped in the class "Datathon_Upload". You can see how you have annotated the dataset by using the "Export" button in the portal.

   - Alternatively, you may use other tools such as **Label Studio** or **INCEpTION**. These rely on offline data and require you to preprocess and import tweet text manually. If you choose this route, please ensure your label set and outputs match the required format for evaluation.

5. Prepare a [X/Twitter dataset](https://github.com/damieh1/datathon_2025/blob/main/create__X_dataset.md), and include a dataset report with label definitions, distribution information, and annotation rationale.

---
### #2 Challenge: Modeling and Evaluation (July 20–27, 2025)

Goal: Use our pre-annotated gold standard dataset to build and evaluate a hate speech detection system.

Participants can select and combine the following curated datasets:
- English data:
   - [Antisemitism on Twitter: A Dataset for Machine Learning and Text Analytics](https://zenodo.org/records/14448399)
   - [Antisemitism on X: Trends in Counter-Speech and Israel-Related Discourse Before and After October 7](https://zenodo.org/records/15025646)

Each dataset consists of user-generated content annotated for antisemitic hate speech and conspiracy narratives. Annotations were created using an expert-reviewed schema and include agreement between multiple annotators.

The labeled English-language datasets contain social media messages from X/Twitter with the keywords "Jews, Israel, Kikes, or Zionazi*," which are classified as either biased or non-biased based on 100% agreement between two annotators.

Tasks:
1. Download the (Goldstandard/GroundTruth) datasets listed above.
2. Use state-of-the-art (SOTA) transformer models to train and fine-tune a system to detect antisemitic content.
3. Evaluate your model:
   - Report precision, recall, F1-score, and display a confusion matrix.
   - List the hyperparameters used for training.
   - Conduct error analysis and provide qualitative examples, especially false positives.

Your code should be submitted in a format that runs on both Unix and Windows systems with clear readme instructions.



## Deliverables

You provide the deliverables in the form of software packages with instructions on how to run them (read-me files). The software must run on any system (Unix or Windows).

The evaluation committee carries out an independent assessment of the teams' performance.

---

**Deliverables for Challenge #1 include:** 
   - Adapting and implementing an existing definition of antisemitism
   - Reporting how the data was scraped and which guidelines were used to classify and annotate the data in a standardized way

Gain **+10 bonus points** by evaluating the consistency of your team’s annotations using an inter-annotator agreement (IAA) metric. This means:
   - Having at least two annotators label a shared subset of the data
   - Calculating a formal agreement score, such as:
      - **Cohen’s Kappa** (for binary or pairwise categorical annotation)
      - **Krippendorff’s Alpha** (especially for multi-class or missing data)

Clearly report:
   - Which subset was double-annotated
   - Your score and a brief interpretation (e.g., “moderate agreement,” “high agreement”)

---

**Deliverables for Challenge #2:**
   - Build and evaluate a transformer-based system for detecting antisemitic content using our pre-annotated gold standard datasets.

1. **Modeling Output**
   - Fine-tune a transformer model (e.g., DeBERTa, RoBERTa, BERTweet, HateBERT) using the provided annotated datasets.
   - Clearly document the model used and include a summary of your training setup (train/test split, random seed, training strategy).

2. **Code Submission**
   - Upload or link your training script(s), configuration files, and any preprocessing pipeline.
   - Your code should run on any standard environment (e.g., Unix or Windows, preferably with Conda or pip environment info).

You can earn up to **+10 bonus points** by testing your model on **unseen data**:

- Collect a small new sample of tweets using Bright Data or another method
- Manually annotate 20–30 examples using the same label scheme
- Apply your trained model to this new set
- Report performance and reflect on how well the model generalizes

You can share the files with us in any way you want. The easiest for you is probably uploading it to any cloud storage and sharing the link with us.

--- 

All deliverables, including annotations and evaluations, must be summarized and completed by **August 5, 2025**.

---
## Prizes & Certificates
- 1st Place Team: $1,000 gift certificate  
- 2nd Place Team: $600 gift certificate  
- 3rd Place Team: $400 gift certificate  

\* Prizes are provided and managed by the World Jewish Congress.

### Participation Certificate

Selected participants will receive a certificate recognizing their achievement and skills.


## Resources and Past Events

- [Tutorial and Material on Social Media Text pre-processing and classification](https://github.com/AnnotationPortal/DatathonandHackathon.github.io/blob/main/NLP_ML_Social_Media_Processing.md)
- [General NLP-Tools, Guides, and Tutorials for all levels](https://github.com/AnnotationPortal/DatathonandHackathon.github.io/blob/c8cc15cf6231e0e994162514d60e4737c34f0cc9/NLP-Tools%20and%20Guides.md)


## Previous material

The material from the 2020 Hackathon and Datathon on Antisemitism in Social Media can be found here:  
[https://github.com/dcavar/AntisemitismDatathon2020](https://github.com/dcavar/AntisemitismDatathon2020)

Information on the 2024 Datathon and Machine Learning Competition can be found here:  
[https://isca.indiana.edu/publication-research/social-media-project/datathon-2024/index.html]

[PPT from January 15, 2023, on psychological and historical reasons for racism and antisemitism](https://github.com/AnnotationPortal/DatathonandHackathon.github.io/blob/main/Psychological%20and%20Historical%20Reasons%20for%20Racism%20and%20Antisemitism.pptx)

[PPT by Damir Cavar, January 29, 2023, Corpus Standards and Evaluation](https://github.com/AnnotationPortal/DatathonandHackathon.github.io/blob/main/Corpus_Format_Selection.pdf) and [Video](https://iu.mediaspace.kaltura.com/media/t/1_5sfcj3ix)

---

<div style="font-size: 0.2em; color: #555; margin-top: 2em;">
  <p>
    All materials in this repository were created by <strong>Daniel Miehling</strong> (Indiana University / ISCA), unless otherwise noted.
  </p>
  <p>
    Please contact <a href="mailto:damieh@iu.edu">damieh@iu.edu</a> for reuse permissions or collaboration inquiries.
  </p>
</div>
