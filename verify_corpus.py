"""Verify corpus composition against proposal requirements"""
from collections import Counter
import json

print("=" * 70)
print("CORPUS COMPOSITION ANALYSIS")
print("=" * 70)

sources = Counter()
directories = Counter()

with open("data/indexed_corpus.jsonl") as f:
    for i, line in enumerate(f):
        if i % 5000 == 0:
            print(f"  Scanned {i} documents...", end='\r')
        row = json.loads(line)
        sources[row["metadata"]["source"]] += 1
        directories[row["metadata"]["directory"]] += 1

print(f"\nTotal documents: {sum(sources.values())}")
print("\n📊 Source Distribution:")
for source, count in sources.most_common():
    pct = (count / sum(sources.values())) * 100
    print(f"  {source:20s}: {count:6d} ({pct:5.1f}%)")

print(f"\n📂 Directory Distribution (top 10):")
for directory, count in directories.most_common(10):
    pct = (count / sum(directories.values())) * 100
    print(f"  {directory:30s}: {count:6d} ({pct:5.1f}%)")

# Check proposal alignment
proposal_sources = {"PubMed", "WHO", "CDC", "MedQuAD"}
found_sources = set(sources.keys())

print(f"\n✓ Proposal Alignment Check:")
for ps in proposal_sources:
    if ps in found_sources:
        print(f"  ✅ {ps}: {sources[ps]} documents")
    else:
        print(f"  ⚠️  {ps}: NOT FOUND")

missing = proposal_sources - found_sources
if missing:
    print(f"\n⚠️  PROPOSAL GAP: Missing sources: {', '.join(missing)}")
    print("    Consider ingesting these sources before publication")
else:
    print(f"\n✅ ALL PROPOSAL SOURCES PRESENT")
