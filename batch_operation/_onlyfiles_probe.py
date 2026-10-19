"""Dry probe: how many Calibre books match only_files whitelist (no DB writes)."""
import asyncio
import os, sys
sys.path.insert(0, "/app")
os.chdir("/app")
from jindai.__main__ import _init_plugins

_init_plugins()
from jindai.pipeline import PipelineStage
from jindai.storage import storage
from plugins.bibliography.datasources.calibre import CalibreDataSource

WL = "/app/batch_operation/only_files_lt600_nondict.txt"


async def main():
    ds = CalibreDataSource()
    ds.log = lambda *x: print("log:", " ".join(str(i) for i in x))
    ds.apply_params(dataset_name="书库", lang="auto", content="OneDrive/",
                    formats="pdf", scan_for_moved=True, only_files=WL)
    paths = await PipelineStage.parse_paths(ds.paths)
    print("library paths:", paths)
    total = 0
    for p in paths:
        raw = ds.get_calibre_books_safe(storage.safe_join(p))
        pdfs = [b for b in raw if b["file_path"].lower().endswith(ds.formats)]
        matched = [b for b in pdfs if b["file_path"] in ds.only_files]
        print(f"  {p}: raw={len(raw)} pdf={len(pdfs)} matched={len(matched)}")
        total += len(matched)
    print("TOTAL matched:", total, "| listed:", len(ds.only_files))
    # show a few listed-but-unmatched to catch path-format mismatch
    listed = list(ds.only_files)[:3]
    print("listed sample:", listed)


asyncio.run(main())
