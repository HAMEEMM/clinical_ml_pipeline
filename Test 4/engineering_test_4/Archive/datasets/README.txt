Deterministic Synthetic Datasets (Seed=42)

1) patient_readmission.csv
   - 10,000 rows; binary label readmitted_30d (~18% positives)
   - Columns:
     patient_id (int), age (int), bmi (float), systolic_bp (int), diastolic_bp (int),
     hba1c (float, may be NaN), prior_admissions (int), smoker (0/1), sex (0/1),
     sdoh_index (float, may be NaN), notes_length (int), readmitted_30d (0/1)

2) patient_readmission_shifted.csv
   - Same schema; distribution + label shift for drift detection.

3) rag_docs.csv
   - 300 short docs (doc_id, title, text)

4) rag_qa.csv
   - 30 Q&A pairs (q_id, question, answer, gt_doc_ids [space-separated top3])

5) doc_embeddings.npy / query_embeddings.npy
   - float32 arrays (300x384) and (30x384), L2-normalized.
   - These embeddings are deterministic TF-IDF+SVD projections (not external models).

6) rag_ground_truth_top5.csv
   - Deterministic top-5 nearest doc_ids per query (cosine over provided embeddings).
