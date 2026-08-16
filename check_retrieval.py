from agent import retrieve

results = retrieve("Can I use Nimbus Notes without an internet connection?", k=10)
for r in results:
    print(round(r['score'], 4), r['source'], '|', r['text'][:80].replace('\n', ' '))
