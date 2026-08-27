#!/usr/bin/env python3
"""
Analyze HippoRAG 2 missed passages on DocHop-QA (n=500 queries).
Since passage text is not in the provided JSON, classify by document metadata
(PMC source documents) and query type.
"""

import json
import re
from collections import defaultdict, Counter

# Load data
with open('../hipporag_results/retrieval.json') as f:
    retrieval_data = json.load(f)

with open('../hipporag_results/gold_n500.json') as f:
    gold_data = json.load(f)

with open('../hipporag_results/eval_n500.json') as f:
    eval_data = json.load(f)

# retrieval.json: queries is a list of query records
retrieval_queries_list = retrieval_data.get('queries', [])
print(f"Total queries in retrieval: {len(retrieval_queries_list)}")

# Build gold_metadata map (qid -> metadata)
gold_metadata = {}
for qdata in eval_data['per_query']:
    qid = str(qdata['qid'])
    gold_metadata[qid] = {
        'query': qdata.get('query', ''),
        'gold_docs': qdata.get('top5_pmcs', []),
        'gold_sections': qdata.get('top5_sids', []),
        'n_gold_sections': qdata.get('n_gold_sections', 0)
    }

# Also merge with gold_n500 metadata
for qid_str, qmeta in gold_data.items():
    if qid_str in gold_metadata:
        gold_metadata[qid_str]['query_type'] = qmeta.get('type', 'unknown')
        gold_metadata[qid_str]['hop_count'] = qmeta.get('hop', 0)

# Build retrieval sets and gold sets
retrieval_sets = {}
gold_pids = {}

for query_record in retrieval_queries_list:
    qid = str(query_record.get('qid', ''))

    # Gold passages from retrieval.json
    gold_pids[qid] = set(str(p) for p in query_record.get('gold_paragraph_ids', []))

    # Retrieved passages from results
    results = query_record.get('results', [])
    retrieved_pids = set()
    for i, result in enumerate(results):
        if i >= 200:
            break
        if isinstance(result, dict) and 'chunk_id' in result:
            retrieved_pids.add(str(result['chunk_id']))
    retrieval_sets[qid] = retrieved_pids

print(f"Queries processed: {len(retrieval_sets)}")

# Find missed passages
missed_passages = {}
missed_count_total = 0
found_count_total = 0

for qid in gold_pids:
    if qid not in retrieval_sets:
        continue

    gold = gold_pids[qid]
    retrieved = retrieval_sets[qid]
    missed = gold - retrieved
    found = gold & retrieved

    found_count_total += len(found)
    missed_count_total += len(missed)

    for pid in missed:
        if pid not in missed_passages:
            missed_passages[pid] = []
        missed_passages[pid].append(qid)

print(f"Total gold passage instances: {found_count_total + missed_count_total}")
print(f"Retrieved instances (k<=200): {found_count_total}")
print(f"Missed instances: {missed_count_total}")
print(f"Distinct missed passages: {len(missed_passages)}")

if found_count_total + missed_count_total > 0:
    recall = found_count_total / (found_count_total + missed_count_total)
    print(f"Recall@200: {recall:.4f}")
else:
    recall = 0

# Classify missed passages by document type (PMC biomedical domain analysis)
# Extract document ID from chunk_id (format: doc_id||chunk_id)
def get_doc_id(chunk_id):
    if '||' in str(chunk_id):
        return str(chunk_id).split('||')[0]
    return str(chunk_id)

def classify_by_query_type(qid):
    """Classify missed passages by the type of query they appear in."""
    meta = gold_metadata.get(qid, {})
    return meta.get('query_type', 'unknown')

# Aggregate missed passages by query type (proxy for entity type)
missed_by_query_type = defaultdict(list)
query_type_counter = Counter()

for pid, qids in missed_passages.items():
    # Get dominant query type across queries where this passage was missed
    types = [classify_by_query_type(qid) for qid in qids]
    type_counter = Counter(types)
    dominant_type = type_counter.most_common(1)[0][0] if type_counter else 'unknown'

    missed_by_query_type[dominant_type].append({
        'pid': pid,
        'qids': qids,
        'n_misses': len(qids)
    })
    query_type_counter[dominant_type] += 1

# Print category breakdown
print("\n=== MISSED PASSAGES BY QUERY TYPE ===")
for qtype, count in query_type_counter.most_common():
    pct = 100.0 * count / len(missed_passages) if missed_passages else 0
    print(f"{qtype:20s}: {count:4d} ({pct:5.1f}%)")

# Extract examples
print("\n=== EXAMPLES PER QUERY TYPE ===")
for qtype, count in query_type_counter.most_common():
    print(f"\n{qtype}:")
    for ex in missed_by_query_type[qtype][:2]:
        sample_qid = ex['qids'][0]
        query_text = gold_metadata.get(sample_qid, {}).get('query', '')[:100]
        print(f"  PID {ex['pid']} (missed in {ex['n_misses']} queries)")
        print(f"    Sample query: {query_text}...")

# Save detailed results
examples_by_type = {}
for qtype, _ in query_type_counter.most_common():
    examples_by_type[qtype] = [
        {'pid': ex['pid'], 'n_misses': ex['n_misses']}
        for ex in missed_by_query_type[qtype][:3]
    ]

with open('../missed_passages_analysis.json', 'w') as f:
    json.dump({
        'total_missed': missed_count_total,
        'distinct_missed': len(missed_passages),
        'total_retrieved': found_count_total,
        'recall_k200': recall,
        'query_type_counts': dict(query_type_counter),
        'examples_by_type': examples_by_type
    }, f, indent=2)

print("\nAnalysis saved to ../missed_passages_analysis.json")
