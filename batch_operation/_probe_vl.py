#!/usr/bin/env python3
# 试另一个免费视觉通道能否读署名（不写库）
import json, re, ssl, base64, csv, urllib.request, fitz
cfg=open("/home/zhuth/.hermes/profiles/collector/config.yaml",encoding="utf-8").read()
base,key=re.search(r"newapi-local:\s*\n\s*api:\s*(\S+)\s*\n\s*api_key:\s*(\S+)",cfg).groups()
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
B="/home/zhuth/jindai-lab/jindai/batch_operation/reports/"
STORE="/misc/Public/jindai-storage"
rows=list(csv.DictReader(open(B+"latin_surname_rows.tsv",encoding="utf-8"),delimiter="\t"))
done=str({json.loads(l).get("id") for l in open(B+"latin_vision.jsonl",encoding="utf-8")})
pick=[r for r in rows if str(r.get("id")) not in done and r.get("path") and r["path"].endswith(".pdf")][:1]
r0=pick[0]
print("取件:", r0.get("authors"), "|", r0["path"][-60:])
import os
raw=r0["path"]
raw=raw[7:] if raw.startswith("../app/") else raw
pp=os.path.join(STORE, raw)
print("存在:", os.path.exists(pp))
doc=fitz.open(pp); b64=base64.b64encode(doc[0].get_pixmap(dpi=110).tobytes("png")).decode(); doc.close()
msg=[{"role":"user","content":[{"type":"text","text":"Read the author name(s) on this page. Reply ONLY JSON {\"authors\":[...]}"},
     {"type":"image_url","image_url":{"url":"data:image/png;base64,"+b64}}]}]
for mdl in ["inclusionai/ling-3.0-flash-vl:free","google/gemma-4-26b-a4b-it:free","qwen3.8-omni-flash"]:
    body=json.dumps({"model":mdl,"messages":msg,"max_tokens":300}).encode()
    req=urllib.request.Request(base.rstrip("/")+"/chat/completions",data=body,
        headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
    try:
        d=json.loads(urllib.request.urlopen(req,timeout=90,context=ctx).read())
        print("  %-42s OK -> %s" % (mdl, d["choices"][0]["message"]["content"][:110].replace("\n"," ")))
    except urllib.error.HTTPError as e:
        print("  %-42s HTTP %s -> %s" % (mdl, e.code, e.read()[:110]))
    except Exception as e:
        print("  %-42s %s %s" % (mdl, type(e).__name__, e))
