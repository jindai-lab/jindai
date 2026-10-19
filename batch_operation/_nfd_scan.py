
import asyncio, sys, unicodedata, os
sys.path.insert(0, "/app")
import common as C

def has_nfd(s):
    return any(unicodedata.combining(c) for c in s)

async def go():
    conn = await C.get_conn()
    try:
        rows = await conn.fetch("SELECT id, path FROM file_metadata WHERE path LIKE 'OneDrive/%' OR path LIKE 'Authors/%' OR path LIKE 'Supplementary Books/%'")
    finally:
        await conn.close()
    nfd = [(str(r["id"]), r["path"]) for r in rows if has_nfd(r["path"])]
    print("NFD行:", len(nfd))
    for i,(i_,p) in enumerate(nfd[:40]):
        nfc = unicodedata.normalize("NFC", p)
        disk_nfc = os.path.exists("/storage/"+nfc)
        disk_orig = os.path.exists("/storage/"+p)
        print(repr(p)[:80], "| NFC在盘:", disk_nfc, "| 原路径在盘:", disk_orig, "| NFC==原:", nfc==p)
    import json
    json.dump([x[1] for x in nfd], open("/app/batch_operation/reports/nfd_paths.json","w"), ensure_ascii=False)
asyncio.run(go())
