*Details about our Streamlit app that we deploy on Hugging Face Space:*

---
title: AI Exposure Job Transition Recommender
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

This Streamlit app helps users explore career transitions to occupations
with lower AI exposure based on skill similarity and exposure scores.

Built using O*NET occupation data and AI exposure scores calculated by ILO.

---
*The way we build our project:*
---
PROJECT TITLE: Career Transitions and Skill Requirements in an AI-Exposed Labour Market
Remember to install all the packages from requirements.txt

1. To create our project we downloaded file that contains all occupations with their SOC Code from ONET Webbsite and named it Occupations.csv
2. In the file Crosswalking.ipynb we loaded Occupations.csv to inspect it
3. After loaded Occupations.csv, 2010_to_2019_Crosswalk.csv and ISCO_SOC_Crosswalk.csv
4. OUTPUT: Occupations_with_SOC2010_ISCO.csv
5. INPUT: ISCO 4digit CODE with AI-exposure score.csv and Occupations_with_SOC2010_ISCO.csv
6. OUTPUT: ILO_Occupations_with_exposure_score.csv - after crosswalking, file that contains ONET occupations with ILO AI exposure scores assigned
7. File "Mean_creation.ipynb". INPUT: ILO_Occupations_with_exposure_score.csv
8. OUTPUT: Occupations_unique_cleaned.csv - cleand file with averaged exposure score
9. File: "LLM_Description.ipynb". INPUT: O*NET_Skills_Tasks_Description.csv
10. Extraction output: ExtractedSummaries.csv
11. File: "embeddings(SBERT).ipynb" INPUT: ExtractedSummaries.csv
12. OUTPUT: SBERT_embeddings_summaries.npy, ExtractedSummaries_with_idx.csv
13. File: "cosine_similarity.ipynb" INPUT: SBERT_embeddings_summaries.npy, ExtractedSummaries_with_idx.csv
14. File: "recommender.ipynb". INPUT: ExtractedSummaries_with_idx.csv, SBERT_embeddings_summaries.npy, AI-Exposure_Scores.csv
15. OUTPUT: Occupations_with_summaries_and_exposure.csv
16. File: "Skill_extraction.ipynb". INPUT: clean_occupation_texts.csv
17. OUTPUT: Since we had to run extarction in few parts, all the outputs we got we merged together and stored in a file called: llm_skills_groq_final.csv
18. File: "Taxonomy.ipynb". INPUT: llm_skills_groq_final.csv
19. OUTPUT: skillcanonicalsbert22.csv, soccanonicalsbert22.csv
20. File: "final_recommender.ipynb" INPUT: ExtractedSummaries_with_idx.csv, SBERT_embeddings_summaries.npy, AI-Exposure_Scores.csv, soccanonicalsbert22.csv
21. OUTPUT: ALL_top10_transitions_with_skills.csv
22. FILE: "experiments.ipynb" INPUT: ALL_top10_transitions_with_skills.csv
23. OUTPUT: TOP_missing_skills_frequency.csv, TOP_missing_skills_weighted_by_exposure.csv
24. VIZUALIZATION:
24.1. FILE: resilient_skills_graph.ipynb 
24.2. FILE: plots.ipynb
24.3. FILE: network.ipynb
25. Streamlit: app.py
26. Hugging Face uploading: requirements.txt, app.py, Dockerfile
27. App link: https://huggingface.co/spaces/karob/ai-exposure-recommender

