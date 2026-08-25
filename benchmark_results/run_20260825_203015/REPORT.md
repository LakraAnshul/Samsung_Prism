==================================================
GUIDE WEAVE — STAGE 10 EVALUATION
==================================================

Benchmark Run ID: run_20260825_203015
Benchmark Cases: 5
Model: WA5471ABP
Appliance: Samsung Washing Machine

RETRIEVAL
----------------------------------
Recall@30       100.00%    [diagnostic]
Recall@8        100.00%
MRR             1.0000
nDCG@8          1.0000

ANSWER QUALITY
----------------------------------
Step Recall     74.00%
Step Order      80.00%
Safety          0.00%
Faithfulness    60.00%

IMAGE RETRIEVAL
----------------------------------
Image Recall@3  72.00%

END-TO-END
----------------------------------
Overall Score   75.60%

RERANKER ABLATION
----------------------------------
                     RRF          RRF + Jina   Change      
Recall@8             100.00%      100.00%      +0.00%      
MRR                  0.7667       1.0000       +0.2333     
nDCG@8               0.8262       1.0000       +0.1738     
Step Recall          80.00%       74.00%       -6.00%      
Step Order           80.00%       80.00%       +0.00%      
Safety               33.33%       0.00%        -33.33%     
Faithfulness         69.50%       60.00%       -9.50%      
Image Recall@3       80.00%       72.00%       -8.00%      
Average Latency      21904.8 ms   23664.5 ms   +1759.7 ms  
Recall@30 [diagnostic] 100.00%      100.00%      +0.00%      

==================================================
FAILURE FLAGS & SYSTEM DIAGNOSTICS
----------------------------------
Model Contamination Count:        0
Critical Safety Violation Count:  3
Complete Retrieval Failure Count: 0
Judge Failure Count:              1

==================================================
WORST PERFORMING CASES (DIAGNOSTIC)
==================================================

Top 5 Worst Retrieval Cases:
  1. [case_0001] Query: "How do I fix a washer that does not turn on?"
     Recall@8: 100.00%, nDCG@8: 1.0000
  2. [case_0001] Query: "My Samsung washer won't turn on."
     Recall@8: 100.00%, nDCG@8: 1.0000
  3. [case_0001] Query: "The washing machine cannot turn on properly."
     Recall@8: 100.00%, nDCG@8: 1.0000
  4. [case_0002] Query: "How do I fix a washer that does not start?"
     Recall@8: 100.00%, nDCG@8: 1.0000
  5. [case_0002] Query: "My Samsung washer won't start."
     Recall@8: 100.00%, nDCG@8: 1.0000

Top 5 Worst Answer Quality Cases:
  1. [case_0001] Query: "The washing machine cannot turn on properly."
     Step Recall: 0.00%, Faithfulness: 0.00%
  2. [case_0002] Query: "My Samsung washer won't start."
     Step Recall: 70.00%, Faithfulness: 55.00%
  3. [case_0002] Query: "How do I fix a washer that does not start?"
     Step Recall: 100.00%, Faithfulness: 90.00%
  4. [case_0001] Query: "My Samsung washer won't turn on."
     Step Recall: 100.00%, Faithfulness: 95.00%
  5. [case_0001] Query: "How do I fix a washer that does not turn on?"
     Step Recall: 100.00%, Faithfulness: not_available

Top 5 Worst Image Retrieval Cases:
  1. [case_0001] Query: "The washing machine cannot turn on properly."
     Image Recall@3: 0.00%
  2. [case_0002] Query: "My Samsung washer won't start."
     Image Recall@3: 60.00%
  3. [case_0001] Query: "How do I fix a washer that does not turn on?"
     Image Recall@3: 100.00%
  4. [case_0001] Query: "My Samsung washer won't turn on."
     Image Recall@3: 100.00%
  5. [case_0002] Query: "How do I fix a washer that does not start?"
     Image Recall@3: 100.00%

==================================================