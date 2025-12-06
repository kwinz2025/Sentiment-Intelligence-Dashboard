# Sentiment-Intelligence-Dashboard



### Project Overview



Sentiment-Intelligence-Dashboard is a real-time analytics and machine-learning dashboard built to analyze Google Reviews for 360 The Restaurant at the CN Tower.  

It combines business intelligence with transformer-based sentiment prediction and complaint-theme discovery.



### Dashboard Features

\- KPI metrics (positivity rate, average rating, token length)  

\- Sentiment trends over time  

\- Star rating \& review length distributions  

\- Recent negative reviews (high-risk complaints)  

\- Most liked reviews (hero stories)  

\- Live sentiment prediction using Ensemble 3 (DistilBERT + TinyBERT)  



All insights are powered by the Stage 1 real dataset and a five-stage NLP pipeline involving cleaning, TF-IDF baselines, transformer fine-tuning, topic modeling, and fairness/synthetic augmentation.



### Modeling Pipeline \& Final Ensemble



The project follows a multi-stage machine-learning architecture:



**Stage 1 – Business Cleaning \& EDA**

Cleaned real reviews, created star-based sentiment labels, extracted metadata, and explored customer patterns.



**Stage 2 – TF-IDF Baselines**  

Built traditional machine-learning models to establish baseline accuracy.



**Stage 3 – Transformer Training \& Ensembles**  

Fine-tuned DistilBERT, TinyBERT, and DeBERTa; created generic ensemble inference functions using logit averaging.  

After evaluation on the untouched Stage 1 real test set, the best performer was:



**Final Model: Ensemble 3 (DistilBERT + TinyBERT)**  

\- Strongest macro-F1  

\- Most stable on negative reviews  

\- Fast and memory-efficient for deployment  

\- Used in the Streamlit live predictor  



The dashboard applies this final ensemble to classify new reviews in real time and provide probability-based insights for managers.



