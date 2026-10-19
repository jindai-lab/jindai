"""Fill real authors for the 牛津通识读本VSI rows whose `authors` is the bogus
{"A Very Short Introduction"} (series name mistaken for an author, 2026-10-20).

Evidence, in precedence order, all read from the books' own imported text:
  1. half-title page  "<Author> <TITLE> A Very Short Introduction <n>"   (weight 10)
  2. half-title page  "<TITLE> A Very Short Introduction <n> <Author>"   (author AFTER title)
  3. cover page p0    author line before the ALL-CAPS title              (weight 8)
  4. series list page "available now: <TITLE> <Author>"                  (weight 1, cross-volume)
Manual overrides below are NOT guesses: each was read off a specific page of
that very file (see the EVIDENCE comment on every line).

Matching is done on a "squashed" (letters+digits only, lowercased) copy of the
page text so that OCR variants like "POST- MODERNISM", "THE U.S.CONGRESS",
"NINETEENTH CENTURYBRITAIN" still match their title.

Dry-run by default; --apply writes. Guarded: only rows whose authors still
equal the bogus value are updated.
"""
import asyncio, json, re, sys, logging, argparse, collections

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/batch_operation")
logging.disable(logging.CRITICAL)
import common as C

# title -> author, read off the named page of that very PDF (see module docstring)
MANUAL = {
    "Film Music": ["Kathryn Kalinak"],                     # p7 "1 Kathryn Kalinak Film Music A Very Short Introduction"
    "Philosophy of Science": ["Samir Okasha"],             # series lists of other volumes: "PHILOSOPHY OF SCIENCE Samir Okasha"
    "Ideology": ["Michael Freeden"],                       # series list of Ethics p3: "IDEOLOGY Michael Freeden"
    "Wittgenstein": ["A. C. Grayling"],                    # p1 "A. C. Grayling WITTGENSTEIN A Very Short Introduction"
    "The Russian Revolution": ["S. A. Smith"],             # p5 "S.A.Smith THE RUSSIAN REVOLUTION A Very Short Introduction"
    "The U.S. Congress": ["Donald A. Ritchie"],             # p0 "Donald A Ritchie THE U.S.CONGRESS A Very Short Introduction"
    "Twentieth-Century Britain": ["Kenneth O. Morgan"],    # p5 "Kenneth O. Morgan Twentieth-Century Britain ..."
    "Nineteenth-Century Britain": ["Christopher Harvie", "H. C. G. Matthew"],  # p5
    "The Great Depression and the New Deal": ["Eric Rauchway"],  # p5 "Eric Rauchway The Great Depression & the New Deal ..."
    "Postmodernism": ["Christopher Butler"],               # p5 "Christopher Butler POST- MODERNISM A Very Short Introduction 1"
    "Kierkegaard": ["Patrick Gardiner"],                    # p0/p1 Past Masters "KIERKEGAARD Patrick Gardiner"
    "Buddhism": ["Damien Keown"],                          # Questia front matter "Contributors: Damien Keown - author"
    "Neoliberalism": ["Manfred B. Steger", "Ravi K. Roy"],  # half-title page
    "Schopenhauer": ["Christopher Janaway"],               # p1 "Christopher Janaway SCHOPENHAUER ..."
    "Globalization": ["Manfred Steger"],                   # list pages beat the garbled cover OCR "Mantred B.Stoger"
    "Philosophy": ["Edward Craig"],                        # half-title "Edward Craig Philosophy A Very Short Introduction 1"
    "The Roman Empire": ["Christopher Kelly"],
    "Photography": ["Steve Edwards"],
    "Global Catastrophes": ["Bill McGuire"],
    "Modern Ireland": ["Senia Paseta"],
    "Nelson Mandela": ["Elleke Boehmer"],
    "Progressivism": ["Walter Nugent"],
    "Psychiatry": ["Tom Burns"],
    "Quantum Theory": ["John Polkinghorne"],
    "Rousseau": ["Robert Wokler"],
    "Statistics": ["David J. Hand"],
    "Superconductivity": ["Stephen Blundell"],
    "The United Nations": ["Jussi M. Hanhimaki"],
    "Sexuality": ["Véronique Mottier"],
    "Myth": ["Robert A. Segal"],
    "Renaissance Art": ["Geraldine A. Johnson"],
    "Chaos": ["Leonard A. Smith"],
    "Russell": ["A. C. Grayling"],
    "Biblical Archaeology": ["Eric H. Cline"],
    "Islamic History": ["Adam J. Silverstein"],
    "Memory": ["Jonathan K. Foster"],
    "The Vikings": ["Julian D. Richards"],
    "Nothing": ["Frank Close"],
    "Particle Physics": ["Frank Close"],
    "The History of Life": ["Michael J. Benton"],
    "The Dead Sea Scrolls": ["Timothy H. Lim"],
    "The Tudors": ["John Guy"],
    "The History of Time": ["Leofranc Holford-Strevens"],
    "Archaeology": ["Paul Bahn"],                          # p5 "Paul Bahn Archaeology A Very Short Introduction"
    "Continental Philosophy": ["Simon Critchley"],         # p1 "Simon Critchley CONTINENTAL PHILOSOPHY ..."
    "Democracy": ["Bernard Crick"],                        # p5 "Bernard Crick DEMOCRACY ..."
    "Machiavelli": ["Quentin Skinner"],                    # p0 "Quentin Skinner MACHIAVELLI ..."
    "The Elements": ["Philip Ball"],                       # p5 "Philip Ball THE ELEMENTS ..."
    "Indian Philosophy": ["Sue Hamilton"],                 # p0 "INDIAN PHILOSOPHY A Very Short Introduction Sue Hamilton"
    "Modern Japan": ["Christopher Goto-Jones"],
    "Mormonism": ["Richard Lyman Bushman"],
}

BOGUS = "A Very Short Introduction"

def squash(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())

def title_of(path):
    base = re.sub(r'\.[Pp][Dd][Ff]$', '', path.split('/')[-1])
    return re.sub(r'\s*-\s*A Very Short Introduction\s*$', '', base, flags=re.I).strip()

def squash_spans(sq, sqtitle):
    """start/end (in squash index space) of every occurrence of sqtitle in sq."""
    return [(m.start(), m.end()) for m in re.finditer(re.escape(sqtitle), sq)]

def make_index(txt):
    """map squash-space index -> original-text index"""
    idx = []
    for i, ch in enumerate(txt):
        if ch.isalnum():
            idx.append(i)
    return idx

CAPWORD = re.compile(r"^[A-Z][A-Z\-'’0-9\.]{1,}$")
INITIAL = re.compile(r"^[A-Z]\.$")
STOP = ('a very short', 'very short', 'oxford', 'the series', 'to conceptual',
        'available now', 'new york', 'oup', 'http', 'www', 'questia', 'copyright',
        'margin notes', 'this page', 'contents', 'preface', 'list of', 'for lu',
        'transferred', 'published', 'first published', 'manchester page')

def is_caps_word(w):
    if INITIAL.match(w):
        return False
    bare = re.sub(r'[^A-Za-z]', '', w)
    return len(bare) > 1 and w.isupper()

def extract_name(s, title, max_words=8):
    s = re.sub(r'\s+', ' ', s).strip(' .·,;:|()[]')
    # never let the title itself leak into the name (any case, squashed form too)
    st = squash(title)
    toks = s.split(' ')
    kept = []
    for w in toks[:max_words]:
        bare = re.sub(r"[^A-Za-z'’.\-]", '', w)
        if not bare:
            break
        if kept and is_caps_word(bare):
            break
        kept.append(bare)
        probe = squash(' '.join(kept))
        if st and (probe.endswith(st) or st in probe[len(probe)//2:]):
            kept.pop()
            break
    name = re.sub(r'\s+', ' ', ' '.join(kept)).strip(' .,;:')
    if not name or len(name) < 4 or not re.search(r'[A-Za-z]{3}', name):
        return ''
    if any(name.lower().startswith(x) for x in STOP):
        return ''
    return name

def find_author(txt, title, weight, votes):
    """author sits either right BEFORE or right AFTER the title on this page."""
    if not txt:
        return
    flat = re.sub(r'\s+', ' ', txt).strip()
    if not flat or len(flat) > 400:
        return
    low = flat.lower()
    if 'questia.com/reader' in low or 'page 1 of' in low or 'margin notes' in low:
        return
    idx = make_index(flat)
    sq = squash(flat)
    sqtitle = squash(title)
    if not sqtitle or sqtitle not in sq:
        return
    for a, b in squash_spans(sq, sqtitle):
        if b - a < len(sqtitle) * 0.8:
            continue
        # original-space boundaries
        o_start = idx[a]
        o_end = idx[b - 1] + 1
        before = extract_name(flat[:o_start], title)
        if before:
            votes[before] += weight
        after = extract_name(flat[o_end:], title)
        if after:
            votes[after] += weight * 0.8   # slight penalty: author-before-title is the norm

async def main(apply=False, limit=None):
    conn = await C.get_conn()
    rows = await conn.fetch(
        "SELECT id, path, title, authors::text AS a FROM file_metadata "
        "WHERE authors::text ILIKE '%A Very Short Introduction%' ORDER BY path")
    print("bogus rows:", len(rows))

    # gather front-matter text once
    paths = [r["path"] for r in rows]
    ev = {}
    for i in range(0, len(paths), 300):
        chunk = paths[i:i + 300]
        got = await conn.fetch(
            "SELECT fm.path, p.source_page, p.content FROM file_metadata fm "
            "JOIN paragraph p ON p.source_id = fm.id "
            "WHERE fm.path = ANY($1::text[]) AND p.source_page <= 16", chunk)
        for g in got:
            ev.setdefault(g["path"], {})[g["source_page"]] = g["content"]

    plan = []
    for r in rows:
        t = title_of(r["path"])
        people = MANUAL.get(t)
        src = 'manual' if people else None
        if not people:
            votes = collections.defaultdict(float)
            pages = ev.get(r["path"], {})
            for pg, txt in pages.items():
                flat = ' '.join((txt or '').split())
                if len(flat) <= 190 and re.search(r'Very Short Introduction\s*\d', flat, re.I):
                    find_author(flat, t, 10, votes)
                elif pg == 0:
                    low = (txt or '').lower()
                    if not ('questia' in low or 'publication information' in low) and \
                       not ('university press' in low and len(txt) > 200):
                        find_author(txt, t, 8, votes)
            # cross-volume series lists (weight 1)
            for p2, pages2 in ev.items():
                for pg, txt in pages2.items():
                    if txt and 'available now' in txt.lower():
                        find_author(txt, t, 1, votes)
            if votes:
                best = max(votes.items(), key=lambda kv: kv[1])[0]
                people = [x.strip() for x in re.split(r'\s+(?:and|&)\s+', best) if x.strip()]
                src = 'auto'
        plan.append({"id": str(r["id"]), "path": r["path"], "title": t,
                     "authors": people or [], "src": src})

    unfilled = [p for p in plan if not p["authors"]]
    print("plan: %d rows, %d filled (%d manual), %d UNFILLED"
          % (len(plan), len(plan) - len(unfilled),
             sum(1 for p in plan if p["src"] == 'manual'), len(unfilled)))
    for p in unfilled:
        print("  UNFILLED:", p["path"])

    json.dump(plan, open("/app/batch_operation/reports/vsi_authors_plan.json", "w"),
              ensure_ascii=False, indent=1)

    if apply and not unfilled:
        done = 0
        for p in plan:
            if limit and done >= limit:
                break
            tag = await conn.execute(
                "UPDATE file_metadata SET authors = $1::text[] "
                "WHERE id = $2::uuid AND authors::text ILIKE '%A Very Short Introduction%'",
                p["authors"], p["id"])
            if tag.split()[-1] == '1':
                done += 1
        print("UPDATED", done)
    elif apply and unfilled:
        print("NOT APPLIED: %d rows still unfilled" % len(unfilled))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    asyncio.run(main(apply=a.apply, limit=a.limit))
