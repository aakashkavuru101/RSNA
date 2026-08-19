import re, sys
doc = open('RSNA_knee_roadmap_to_095.md', encoding='utf-8').read()
checks = {
 'CURRENT-LEADERBOARD': all(s in doc for s in ['0.951','0.947','2026-08-19','30% of the test data']),
 'GAP-DIAGNOSIS': 'knowledge' in doc and 'model gap' in doc and '0.83' in doc and '58' in doc,
 'QUANTITATIVE-CEILING': 'Per-label difficulty ranking' in doc and 'Synovitis' in doc,
 'ROADMAP': all(s in doc for s in ['Lever 1','Lever 2','Lever 3','Lever 4','Lever 5','Lever 6']),
 'RISK-CONTROL': 'shake-up' in doc and 'grouped CV' in doc and '70%' in doc,
 'SOURCES': doc.count('http') >= 5 and 'Caveats' in doc,
 'FORMAT': len(doc.split()) > 1500,
}
for k,v in checks.items(): print(('PASS' if v else 'FAIL'), k)
sys.exit(0 if all(checks.values()) else 1)
