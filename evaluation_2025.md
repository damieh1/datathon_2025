
# Evaluation Methodology and Results

The following presents the evaluation methodology and outcomes for the [ISCA Datathon & Machine Learning Competetion](https://github.com/AnnotationPortal/DatathonandHackathon.github.io/tree/main). In total six teams engaged in two complementary challenges: (i) scraping a dataset and apply an annotation framework for antisemitism detection, and (ii) model fine-tuning and evaluation of a goldstandard dataset.

*The scoring framework integrates the official competition rubrics with additional variables related to team composition, prior experience, and dropout rates, thereby incorporating team cohesion and skill retention (capacity) into the ranking-based assessments.*

---

## 1 Evaluation Framework

### 1.1 Data Sources
We assessed teams using:
- Written reports for [Challenge 1](https://github.com/AnnotationPortal/DatathonandHackathon.github.io/tree/main?tab=readme-ov-file#1-challenge-dataset-creation-july-1320-2025) and [Challenge 2](https://github.com/AnnotationPortal/DatathonandHackathon.github.io/tree/main?tab=readme-ov-file#2-challenge-modeling-and-evaluation-july-2027-2025).
- Supplementary materials including code repositories, Colab notebooks, and annotated datasets.
- Official scoring criteria as specified in the competition guidelines.
- Additional variables indicated by participants' prior experience, team cohesion, and skill retention based on dropout rates.
  
---

### 1.2 Official Scoring Criteria

**Challenge #1: Scraping a Dataset & Annotation (50 points total)**

| Variable                  | Max | Definition |
|---------------------------|-----|------------|
| C1_Relevance_Variety      | 10  | Dataset includes relevant antisemitic and non-antisemitic posts; diverse sources (hashtags, keywords, user groups). |
| C1_Annotation_Schema      | 10  | Correct application of IHRA-WDA or justified adaptation; schema maps clearly to labels used. |
| C1_Internal_Consistency   | 10  | Labels applied consistently with minimal misclassification; shows internal review. |
| C1_Data_Report_Quality    | 10  | Dataset report includes keyword/time range, label definitions, label distribution, methodology. |
| C1_Nuance_Reflection      | 10  | Report addresses challenges, limitations, ambiguity, and—if present—social/ethical implications. |
| C1_Bonus_IAA              | 10  | Bonus: Formal inter-annotator agreement (Cohen’s Kappa, Krippendorff’s Alpha) reported and interpreted. |

**Challenge #2: Modeling & Evaluation (50 points total)**

| Variable                  | Max | Definition |
|---------------------------|-----|------------|
| C2_Model_Performance      | 15  | Precision, recall, and F1-score reported; confusion matrix included. |
| C2_Use_Gold_Dataset       | 10  | Used provided gold standard dataset correctly (no data leakage, correct splits). |
| C2_Training_Pipeline      | 10  | Training process documented (hyperparameters, train/test/val split, reproducibility). |
| C2_Error_Analysis         | 10  | Identifies error patterns; gives 3–5 FP/FN examples with reasoning. |
| C2_Documentation          | 5   | Code is clear, well-structured, and reproducible (e.g., runnable Colab/README). |
| C2_Bonus_UnseenData       | 10  | Bonus: Model tested on new, manually annotated unseen data; performance reported and reflected on. |


---
## Addtional Evaluation Variables

### Team composition
    Education: Distinction between high school and undergraduate participants.
    Dropouts: Number of members who left the team during the competition.
    Remaining_Team_Members: Number of members who stayed until the end of the competition.
    Initial_Team_Members: Number of members at the start of the competition.
    
### Coding experience
    Initial_Coding_Experience: Number of initial members with prior coding experience.
    Remaining_Coding_Experience: Number of remaining members with prior coding experience.
    
### Antisemitism knowledge
    Initial_Antisemitism_Knowledge: Number of initial members with prior knowledge about antisemitism.
    Remaining_Antisemitism_Knowledge: Number of remaining members with prior knowledge about antisemitism.
    
### Scores
    Total_Score: Sum of all scoring components from both challenges.
    Weighted_Score: The total score is adjusted based on the proportion of remaining team members, team cohesion, and team capacity.

---

## 3 Evaluation Summary

## Challenge #1: Dataset Creation & Annotation – Top 3 Teams

### Team 6 MagenCode
- Collected **355 tweets** from relevant hashtags and keywords.  
- Applied an annotation scheme aligned with **IHRA-WDA**, labeling posts as antisemitic or not.  
- Report included clear keyword/time range, label definitions, and distribution.  
- Conducted **IAA calculation** (Cohen’s Kappa = 0.54, moderate agreement).  
- Reflected on limitations and challenges in interpreting borderline cases.  

### Team 3 Bias Busters
- Collected a diverse sample of tweets using targeted keywords and user accounts.  
- Adapted IHRA-WDA to their schema and provided justification for category choices.  
- Annotation carried out by all members; no formal IAA reported.  
- Dataset report was transparent, with methodology and label descriptions clearly documented.  
- Included discussion of ambiguity in borderline cases and recommendations for future refinements.  

### Team 2
- Collected tweets from multiple sources using hashtags, keywords, and user handles.  
- Applied IHRA-WDA without modifications.  
- Reported **IAA score**, though agreement was relatively low.  
- Dataset report included methodology, label distribution, and keyword/time range.  
- Reflected on the challenges of labeling implicit antisemitism and the importance of contextual understanding.  

---

## Challenge #2 Transformer Model Performance Summary – Top 3 Teams

**Note:** Teams reported different metrics for model evaluation, and some did not consistently provide all values.  
Missing values are marked as `N/A` and reflect *non-reporting*, not zero or failed results.

| Team | Model Used | Accuracy | Macro F1 | Weighted F1 | Precision | Recall | Class F1 (Not Offensive) | Class F1 (Offensive) | Eval Loss | Train Loss | Notes |
|------|------------|----------|----------|-------------|-----------|--------|--------------------------|----------------------|-----------|------------|-------|
| Team 6 MagenCode | CardiffNLP/twitter-roberta-base-offensive | 90.15% | 0.899 | N/A | N/A | N/A | N/A | N/A | 0.397 | 0.183 | Final reported model (GPU computing) |
| Team 2 | CardiffNLP/twitter-roberta-base-offensive | 87% | 0.58 | 0.85 | N/A | N/A | 0.93 | 0.24 | N/A | N/A | Final reported model |
| Team 3 Bias Busters | CardiffNLP/twitter-roberta-base-hate | N/A | 0.617 | N/A | 0.567 | 0.677 | N/A | N/A | N/A | 0.2107 | Based on best model performance (Epoch 2) |

---

## 4 Results

The table below shows the **final team rankings** for the ISCA Datathon 2025.  
For a detailed explanation of the formulas and methodology, see full [Scoring Methodology](https://damieh1.github.io/evaluation).

| Team   |   Total_Score |   Cohesion_Multiplier |   Capacity_Multiplier |   Final_Score |
|:-------|--------------:|----------------------:|----------------------:|--------------:|
| Team 6 |           119 |                  1.00 |                  1.00 |        119.00 |
| Team 3 |           110 |                  0.75 |                  0.88 |         72.19 |
| Team 2 |           115 |                  0.60 |                  0.67 |         46.00 |
| Team 4 |           117 |                  0.50 |                  0.75 |         43.88 |
| Team 1 |            25 |                  0.20 |                  0.50 |          2.50 |
| Team 5 |             0 |                  0.00 |                  0.00 |          0.00 |

---

## 5 Final Verdict
The top three teams in the weighted-score ranking—**Team 6 MagenCode**, **Team 3 Bias Busters**, and **Team 2**—distinguished themselves through sustained participation and the ability to deliver quality outputs.  

### Placements

- Rank #1: **Team 6 MagenCode**, which retained all four members, demonstrated the highest efficiency, producing a balanced dataset, achieving strong model performance (F1   = 0.899), and completing both challenges with comprehensive documentation.  

- Rank #2: **Team 3 Bias Busters** also maintained high productivity, combining transparent workflows with full challenge completion, despite a small reduction in team       size.  

- Rank #3: **Team 2** began with five members and finished with three, yet they still delivered methodologically solid outputs for both challenges, including unseen data     testing. 

  Although outside the top three in weighted ranking, **Team 4** is notable for producing outputs on par with much larger teams, despite starting and finishing      with only two members.

---
