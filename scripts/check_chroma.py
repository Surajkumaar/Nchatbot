import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.vectorstore import ChromaVectorStore

store = ChromaVectorStore('chromaDB')
store.load()
col = store.collection
print('collection:', col)
try:
    cnt = col.count()
    print('count():', cnt)
except Exception as e:
    print('count() error:', e)
    try:
        resp = col.get(include=['ids','documents','metadatas'])
        print('get keys:', list(resp.keys()))
        ids = resp.get('ids')
        if ids:
            print('ids example len:', len(ids[0]) if isinstance(ids[0], list) else len(ids))
    except Exception as e2:
        print('get error:', e2)

res = store.query('Nursing', top_k=5)
print('query results len:', len(res))
for r in res:
    print('---')
    print('id:', r.get('id'))
    print('distance:', r.get('distance'))
    print('meta:', r.get('metadata'))
    print('doc preview:', (r.get('document') or '')[:200])
