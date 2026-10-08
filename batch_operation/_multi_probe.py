
import asyncio, common as C
async def main():
    conn = await C.get_conn()
    async def texts(path):
        fid = await conn.fetchval("SELECT id FROM file_metadata WHERE path=$1", path)
        rows = await conn.fetch("SELECT content FROM paragraph WHERE source_id=$1::uuid", fid)
        return [r["content"] or "" for r in rows]
    old = await texts('Authors/Rorty, Richard/哲学、文学和政治 - Richard Rorty.pdf')
    t9 = await texts('OneDrive/Richard Rorty/哲学、文学和政治 (9148)/哲学、文学和政治 - Richard Rorty.pdf')
    t4 = await texts('OneDrive/Richard Rorty/哲学、文学和政治 (4833)/哲学、文学和政治 - Richard Rorty.pdf')
    def cmp(a, b):
        sb, sa = set(b), set(a)
        ma = sum(len(x) for x in a if x in sb)
        mb = sum(len(x) for x in b if x in sa)
        return ma, sum(map(len, a)), mb, sum(map(len, b))
    for name, other in (("9148", t9), ("4833", t4)):
        ma, ta, mo, to = cmp(old, other)
        print(f"old vs {name}: old_matched {ma}/{ta} ({ma/max(ta,1):.1%}), other_matched {mo}/{to} ({mo/max(to,1):.1%})")
    await conn.close()
asyncio.run(main())
