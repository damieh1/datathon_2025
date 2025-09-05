# ISCA Datathon 2025 Final Report

The **[ISCA Datathon & Machine Learning Competition on Antisemitism 2025](https://isca.indiana.edu/publication-research/social-media-project/datathon-2025/index.html)**, hosted by the **[Institute for the Study of Contemporary Antisemitism (ISCA), Indiana University](https://isca.indiana.edu/)**, brought together high school and undergraduate students to explore the technical and conceptual challenges of detecting antisemitic hate speech online.  

The Datathon was coordinated by **[Rachel Kelly](https://isca.indiana.edu/about/our-team/faculty-and-staff/3rachel-kelly.html) (Project Manager)**, who managed all correspondence with our partners and social media communications. **[Daniel Miehling](https://damieh1.github.io//) (Computational Research Coordinator)** created and redesigned the challenges and evaluation framework based on iterations of previous Datathons. He also led the [second workshop](https://www.youtube.com/watch?v=EMuQFb-H0CE) on practical methods, including scraping, annotation, coding foundations, and evaluation. **[Günther Jikeli](https://isca.indiana.edu/about/our-team/faculty-and-staff/1jikeli-gunther.html) (ISCA Associate Director)** provided the conceptual framing in the opening session, and **[Damir Cavar](https://damir.cavar.me/)** introduced automated text analysis in [Workshop 3](https://www.youtube.com/watch?v=ERwkQuVQshU).  

We gratefully acknowledge our partners and sponsors — **[The Bright Initiative by Bright Data](https://brightinitiative.com/)**, **[Indiana University](https://indiana.edu/)**, **[World Jewish Congress](https://www.worldjewishcongress.org/)**, **[TECHRI – Technology and Human Rights Institute](https://www.worldjewishcongress.org/en/what-we-do/techri)**, **[Jewish Federation of Greater Indianapolis](https://www.jewishindianapolis.org/)**, and **Diane M. Druck** — all of whom made this competition possible.  

---

## Challenge Overview

### Challenge #1: Dataset Creation & Annotation (July 13–20, 2025)  
Participants used **Bright Data** to scrape posts from [X (formerly Twitter)](https://x.com/), defined sampling strategies, and annotated their datasets using the **[ISCA Annotation Portal](https://annotate.osome.iu.edu/)** or other tools. Annotation followed the **[IHRA Working Definition of Antisemitism (IHRA-WDA)](https://www.holocaustremembrance.com/resources/working-definitions-charters/working-definition-antisemitism)**, with teams allowed to adapt alternative frameworks if they provided clear justifications. Reports included dataset documentation, label definitions, and reflections on ambiguity. Teams could earn bonus points by reporting inter-annotator agreement (IAA).  

### Challenge #2: Modeling & Evaluation (July 20–27, 2025)  
Using ISCA’s **gold-standard antisemitism datasets**, teams fine-tuned transformer models (e.g., [RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-offensive), [DeBERTa](https://huggingface.co/microsoft/deberta-v3-base), [HateBERT](https://huggingface.co/GroNLP/hateBERT), [BERTweet](https://huggingface.co/vinai/bertweet-base)). Submissions included performance metrics, confusion matrices, error analyses, and reproducible code (often in **[Google Colab](https://colab.research.google.com/)**). Bonus points were awarded for testing on newly collected and annotated unseen data.  

---

## Results

### Top 3 Teams

- **Rank #1: Team 6 – MagenCode** *(Oriel Atias, Giulio Zuckermann, Eliana Woolf and Asher Rosenfeld)*
  Retained all four members, created a 355-tweet dataset aligned with IHRA-WDA, reported Cohen’s Kappa = 0.54 (moderate agreement), and fine-tuned RoBERTa to achieve **Macro F1 ≈ 0.899**. Strong methodology and full documentation secured their top placement.  

- **Rank #2: Team 3 – Bias Busters** *(Dvir Sacho-Tanzer, Jacob Neuer and Jennifer Cronkright)*
  Collected a diverse dataset, adapted IHRA-WDA with justification, and documented methodology transparently. Their RoBERTa-hate model achieved **Macro F1 ≈ 0.617**. Despite reduced team size, they completed both challenges comprehensively.  

- **Rank #3: Team 2 – Code4Clarity** *(Syed Afnan Adit, Mark Vinokur and Saisha Siram)*
  Applied IHRA-WDA directly, reported IAA (low but transparent), and built a well-structured dataset. Their RoBERTa-offensive model achieved **Accuracy ≈ 87%**, with clear error analysis and even unseen-data testing.  

### Honorable Mention  

- **Team 4** *(Dena Shink and Daniel Macholl)* delivered high-quality results comparable to larger teams despite working with only two members throughout the Datathon.  

---

## Congratulations

We extend our warm congratulations to **all participants** — high school and undergraduate students — for their creativity, dedication, and hard work. The ISCA Datathon 2025 showcased how young researchers can meaningfully combine conceptual frameworks and computational tools to confront antisemitism online.  

Through this year’s competition, students not only gained experience in **annotation, dataset creation, and model evaluation**, but also learned to critically reflect on the limits and responsibilities of automated hate-speech detection.  
