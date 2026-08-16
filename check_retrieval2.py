from agent import retrieve

for q in [
    "Can I use Nimbus Notes without an internet connection?",
    "How do I export my notes and what formats are supported?",
]:
    print("=" * 80)
    print("QUERY:", q)
    results = retrieve(q, k=10)
    for r in results:
        print("-" * 40)
        print(round(r['score'], 4), r['source'])
        print(r['text'])
