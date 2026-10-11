"""author_unify.py — 同人异写合并 + 外国作者译名→原名（dry-run 默认，2026-10-11）。

证据分层（只认**同一件作品**里的署名，逐层更硬）：

  C1 Calibre   OneDrive/<书名> (N)/ 取**最后一个** `(N)` 当 book id（`(7)` 是卷号，
               取第一个会拿错书——实测 18189 被读成 7 就是这么来的）→
               books_authors_link 里若有人名拉丁的著者，就用拉丁名替换行内的中文名。
               Calibre 常见写法 `['Ludwig Wittgenstein', '楼巍']`（原名 + 译者），
               正好符合「authors = 著者」的口径：取拉丁、丢译者。
  D1 目录/文件名 作者目录第 2 段，或文件名 `<题名> - <作者>` 的作者段是拉丁人名
               （且题名段不是拉丁人名，避免把 `Saving Beauty` 当作者）。
  Z1 Zotero    行带外国标记（〔德〕（英）[法] / authors 的 (国籍) 尾巴）且题名**精确**
               命中单个拉丁著者。
  M1 人工表    CURATED（国别后缀 40 行 + 高确信的通译名）。
  U1 异写合并  以上替换完成后按归一化键聚簇取规范形（西方人→原文名；同语种仅换序不并；
               簇上限 5；未知/佚名 永不当簇心）。
  X1 补 未知   authors 是 `未知` 而 Calibre/目录给了拉丁著者 —— 单独成档，默认不写。

dry-run 默认；--apply 写库（先落 reports/authors_unify_backup_<date>.tsv，
UPDATE 用 authors::text 旧值护栏）。
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C  # noqa: E402

log = C.setup_logging("author_unify", "batch_operation/reports/authors_unify.log")

LATINISH = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ .'\-]*$")
CJK_PURE = re.compile(r"^[一-鿿·、，]+$")
NATION_TAIL = re.compile(r"[（(][一-鿿]{1,6}[)）]\s*$")
NATION_MARK = re.compile(
    r"[〔\[（(]\s*(?:美|英|法|德|日|俄|意|西|奥|苏|荷兰|比利时|瑞典|挪威|丹麦|芬兰|波兰|"
    r"捷克|匈牙利|希腊|葡萄牙|瑞士|以色列|加拿大|澳大利亚|印度|韩国|朝鲜|墨西哥|保加利亚|"
    r"塞尔维亚|格鲁吉亚|智利|黎巴嫩)\s*[\]）)]")
KANA = re.compile(r"[぀-ヿ]")
HANGUL = re.compile(r"[가-힯]")
JP_NAMES = {"東浩紀": "东浩纪", "伊藤瑞子": "伊藤瑞子", "富井玲子": "富井玲子",
            "铃木康广": "铃木康广", "北中淳子": "北中淳子", "土佐尚子": "土佐尚子"}
LATIN_CANON = {"HENRY DAVID THOREAU": "Henry David Thoreau",
               "Slavoj Zizek": "Slavoj Žižek"}
JUNK = {"未知", "佚名", "无名", "无作者", "不详", "unknown", "none",
        "n.a.", "na", "其他", ""}

# ------------------------------------------------------------ 人工表（高确信）
CURATED: dict[str, str] = {
    # 国别后缀那 40 行
    "里尔克": "Rainer Maria Rilke", "卡夫卡": "Franz Kafka",
    "茨威格": "Stefan Zweig", "艾特玛托夫": "Chingiz Aitmatov",
    "波爱修斯": "Boethius", "波爱修": "Boethius", "奥维德": "Ovid",
    "塞涅卡": "Seneca", "维吉尔": "Virgil", "西塞罗": "Cicero",
    "卡图卢斯": "Catullus", "塞万提斯": "Miguel de Cervantes",
    "卡尔德隆": "Pedro Calderón de la Barca", "维加": "Lope de Vega",
    "纪伯伦": "Kahlil Gibran", "格拉宁": "Daniil Granin",
    "瓦普察洛夫": "Nikola Vaptsarov", "米洛拉德·帕维奇": "Milorad Pavić",
    "尼可斯·卡赞扎基斯": "Nikos Kazantzakis",
    "考琳·麦卡洛": "Colleen McCullough", "阿·托尔斯泰": "Aleksey Tolstoy",
    "伯恩哈德": "Thomas Bernhard", "鲁斯塔维里": "Shota Rustaveli",
    "叶夫图申科": "Yevgeny Yevtushenko",
    # 上一轮按题名封面填入的译名
    "丘特切夫": "Fyodor Tyutchev", "洛尔卡": "Federico García Lorca",
    "博纳富瓦": "Yves Bonnefoy", "奈保尔": "V. S. Naipaul",
    "叔本华": "Arthur Schopenhauer", "埃德蒙·伯克": "Edmund Burke",
    "阿格农": "S. Y. Agnon", "马尔库塞": "Herbert Marcuse",
    # 通译名（无歧义、库里高频）
    "黑塞": "Hermann Hesse", "韩炳哲": "Byung-Chul Han",
    "宇文所安": "Stephen Owen", "克拉里": "Jonathan Crary",
    "蒙田": "Michel de Montaigne", "马克思": "Karl Marx",
    "恩格斯": "Friedrich Engels", "黑格尔": "G. W. F. Hegel",
    "康德": "Immanuel Kant", "尼采": "Friedrich Nietzsche",
    "海德格尔": "Martin Heidegger", "维特根斯坦": "Ludwig Wittgenstein",
    "哈贝马斯": "Jürgen Habermas", "阿多诺": "Theodor W. Adorno",
    "本雅明": "Walter Benjamin", "布洛赫": "Ernst Bloch",
    "卢卡奇": "György Lukács", "齐泽克": "Slavoj Žižek",
    "福柯": "Michel Foucault", "德里达": "Jacques Derrida",
    "德勒兹": "Gilles Deleuze", "瓜塔里": "Félix Guattari",
    "罗兰·巴特": "Roland Barthes", "巴塔耶": "Georges Bataille",
    "鲍德里亚": "Jean Baudrillard", "利奥塔": "Jean-François Lyotard",
    "列维纳斯": "Emmanuel Levinas", "梅洛-庞蒂": "Maurice Merleau-Ponty",
    "阿尔都塞": "Louis Althusser", "布尔迪厄": "Pierre Bourdieu",
    "萨特": "Jean-Paul Sartre", "波伏娃": "Simone de Beauvoir",
    "加缪": "Albert Camus", "列维-斯特劳斯": "Claude Lévi-Strauss",
    "巴赫金": "Mikhail Bakhtin", "陀思妥耶夫斯基": "Fyodor Dostoevsky",
    "托尔斯泰": "Leo Tolstoy", "果戈理": "Nikolai Gogol",
    "屠格涅夫": "Ivan Turgenev", "契诃夫": "Anton Chekhov",
    "高尔基": "Maxim Gorky", "普希金": "Alexander Pushkin",
    "莱蒙托夫": "Mikhail Lermontov", "帕斯捷尔纳克": "Boris Pasternak",
    "索尔仁尼琴": "Aleksandr Solzhenitsyn", "纳博科夫": "Vladimir Nabokov",
    "波德莱尔": "Charles Baudelaire", "兰波": "Arthur Rimbaud",
    "瓦莱里": "Paul Valéry", "普鲁斯特": "Marcel Proust",
    "纪德": "André Gide", "马拉美": "Stéphane Mallarmé",
    "歌德": "Johann Wolfgang von Goethe", "席勒": "Friedrich Schiller",
    "海涅": "Heinrich Heine", "卡尔维诺": "Italo Calvino",
    "艾柯": "Umberto Eco", "但丁": "Dante Alighieri",
    "毕加索": "Pablo Picasso", "康定斯基": "Wassily Kandinsky",
    "杜尚": "Marcel Duchamp", "罗丹": "Auguste Rodin",
    "麦克卢汉": "Marshall McLuhan", "阿伦特": "Hannah Arendt",
    "桑塔格": "Susan Sontag", "葛兰西": "Antonio Gramsci",
    "阿甘本": "Giorgio Agamben", "奈格里": "Antonio Negri",
    "哈特": "Michael Hardt", "拉康": "Jacques Lacan",
    "弗洛伊德": "Sigmund Freud", "荣格": "Carl Jung",
    "波普尔": "Karl Popper", "库恩": "Thomas Kuhn",
    "维纳": "Norbert Wiener", "图灵": "Alan Turing",
    "吉登斯": "Anthony Giddens", "杰姆逊": "Fredric Jameson",
    "萨义德": "Edward Said", "哈维": "David Harvey",
    "列斐伏尔": "Henri Lefebvre", "伊格尔顿": "Terry Eagleton",
    "威廉斯": "Raymond Williams", "霍尔": "Stuart Hall",
    "德·塞尔托": "Michel de Certeau", "德赛尔托": "Michel de Certeau",
    "巴什拉": "Gaston Bachelard", "康吉莱姆": "Georges Canguilhem",
    "马舍雷": "Pierre Macherey", "罗蒂": "Richard Rorty",
    "马尔库塞": "Herbert Marcuse",
    "奎因": "W. V. O. Quine", "塞尔": "John Searle",
    "杜威": "John Dewey", "詹姆斯": "William James",
}

# 全名译名（人名含「·」或带国别前缀）—— 必须**精确**命中，绝不做「以姓结尾」的模糊匹配
# （实测教训：'亨利·詹姆斯' 被判成 '威廉·詹姆斯' 的 William James，实为 Henry James）
CURATED_FULL: dict[str, str] = {
    "托马斯·哈代": "Thomas Hardy", "简·奥斯丁": "Jane Austen",
    "米兰·昆德拉": "Milan Kundera", "阿兰·德波顿": "Alain de Botton",
    "艾米莉·勃朗特": "Emily Brontë", "夏洛蒂·勃朗特": "Charlotte Brontë",
    "E·B·怀特": "E. B. White", "亨利·詹姆斯": "Henry James",
    "戴维·洛奇": "David Lodge", "托马斯·曼": "Thomas Mann",
    "斯坦尼斯拉夫·莱姆": "Stanisław Lem", "杰克·伦敦": "Jack London",
    "约翰·福尔斯": "John Fowles", "艾尔莎·莫兰黛": "Elsa Morante",
    "辛克莱·刘易斯": "Sinclair Lewis", "迈克尔·赫兹菲尔德": "Michael Herzfeld",
    "阿利桑德罗·巴里科": "Alessandro Baricco", "雅克·朗西埃": "Jacques Rancière",
    "马克·吐温": "Mark Twain", "D·H·劳伦斯": "D. H. Lawrence",
    "T·S·艾略特": "T. S. Eliot", "马丁·海德格尔": "Martin Heidegger",
    "斯拉沃热·齐泽克": "Slavoj Žižek", "马歇尔·麦克卢汉": "Marshall McLuhan",
    "埃德蒙·伯克": "Edmund Burke", "考琳·麦卡洛": "Colleen McCullough",
    "米洛拉德·帕维奇": "Milorad Pavić", "尼可斯·卡赞扎基斯": "Nikos Kazantzakis",
    "阿·托尔斯泰": "Aleksey Tolstoy", "西尔维·库尔廷·德纳米": "Sylvie Courtine-Denamy",
    "迈克尔·哈特": "Michael Hardt", "菲利克斯·加塔利": "Félix Guattari",
    "乔治·吉辛": "George Gissing", "乔治·奥威尔": "George Orwell",
    "乔治·艾略特": "George Eliot", "丹尼尔·凯尔曼": "Daniel Kehlmann",
    "伊凡·克里玛": "Ivan Klíma", "伊曼努尔·列维纳斯": "Emmanuel Levinas",
    "保罗·维利里奥": "Paul Virilio", "克莱尔·毕夏普": "Claire Bishop",
    "克里斯蒂安·福克斯": "Christian Fuchs", "列夫·马诺维奇": "Lev Manovich",
    "劳拉·贝茨": "Laura Bates", "卡罗·勒维": "Carlo Levi",
    "史蒂芬·都伯纳": "Stephen Dubner", "哈罗德·罗森伯格": "Harold Rosenberg",
    "大卫·米切尔": "David Mitchell", "奥尔罕·帕慕克": "Orhan Pamuk",
    "安·兰德": "Ayn Rand", "安·拉莫特": "Anne Lamott",
    "安·比尔斯": "Ambrose Bierce", "安东尼·伯吉斯": "Anthony Burgess",
    "安德鲁·派珀": "Andrew Piper", "尤瓦尔·赫拉利": "Yuval Noah Harari",
    "弗里德里希·A.基特勒": "Friedrich A. Kittler", "恩斯特·格罗塞": "Ernst Grosse",
    "托妮·莫里森": "Toni Morrison", "托马斯·品钦": "Thomas Pynchon",
    "托马斯·沃尔夫": "Thomas Wolfe", "拉·艾里森": "Ralph Ellison",
    "本杰明·富兰克林": "Benjamin Franklin", "朱迪思·科克": "Judith Koke",
    "柯蒂斯·L.卡特": "Curtis L. Carter", "格奥尔格·卢卡奇": "György Lukács",
    "格雷厄姆·格林": "Graham Greene", "格雷戈里·贝特森": "Gregory Bateson",
    "楚曼·卡波第": "Truman Capote", "沃伦·蒙塔格": "Warren Montag",
    "海因里希·伯尔": "Heinrich Böll", "爱德华·A.尚肯": "Edward A. Shanken",
    "特奥多尔·W.阿多诺": "Theodor W. Adorno", "玛格丽特·劳伦斯": "Margaret Laurence",
    "理查德·奥斯曼": "Richard Osman", "瑞·达利欧": "Ray Dalio",
    "甘诺特·波梅": "Gernot Böhme", "科尔姆·托宾": "Colm Tóibín",
    "约卡伊·莫尔": "Mór Jókai", "约斯·德·穆尔": "Jos de Mul",
    "约翰·福尔克": "John H. Falk", "约翰·阿米蒂奇": "John Armitage",
    "约翰娜·德鲁克": "Johanna Drucker", "罗伯特·M·波西格": "Robert M. Pirsig",
    "罗伯特·S.纳尔逊": "Robert S. Nelson", "罗曼·罗兰": "Romain Rolland",
    "艾丽斯·沃克": "Alice Walker", "艾伦·刘": "Alan Liu",
    "艾斯特哈兹·彼得": "Péter Esterházy",
    "莫伊塞斯·贝拉斯克斯-曼诺夫": "Moisés Velasquez-Manoff",
    "莱因哈特·布兰特": "Reinhard Brandt", "菲利普·奥斯兰德": "Philip Auslander",
    "菲利普·罗斯": "Philip Roth", "费尔南多·波亚托斯": "Fernando Poyatos",
    "赫尔穆特·普勒斯纳": "Hellmuth Plessner", "阿格妮丝·赫勒": "Agnes Heller",
    "雷沙德·卡普希钦斯基": "Ryszard Kapuściński", "雷蒙德·卡佛": "Raymond Carver",
    "霍伊特·朗": "Hoyt Long", "霍马·卡图赞": "Homa Katouzian",
    "马修·兰普利": "Matthew Rampley", "马茨·弗里德隆": "Mats Fridlund",
    "鲍里斯·维昂": "Boris Vian", "路易斯·P·波伊曼": "Louis P. Pojman",
    "路罗斯·P·波伊曼": "Louis P. Pojman", "欧文·M·柯匹": "Irving M. Copi",
    "卡尔·柯恩": "Carl Cohen", "布鲁斯·昂": "Bruce Aune",
    "约翰·海尔": "John Heil", "肯达尔·L.沃尔顿": "Kendall L. Walton",
    "杰弗里·托马斯": "Geoffrey Thomas",
    "西蒙娜·德·波伏瓦": "Simone de Beauvoir",
    "伊丹·本-巴拉克": "Idan Ben-Barak", "卡梅伦·韦斯特": "Cameron West",
    "约翰娜· 德鲁克 朱子彤／译": "Johanna Drucker",
    # 联网取证（2026-10-11）
    "亚历山大杰弗里·查尔斯": "Jeffrey C. Alexander",   # 《社会》2015 英文署名 Jeffrey Charles Alexander
    "斯蒂文·拉格斯": "Steven Ruggles",               # 《数字人文研究》2021，明尼苏达人口中心
    "朱利安·胡斯": "Julian Huss",                   # 《上海视觉》2021 英文署名
    "迪弗蒂埃里·德": "Thierry de Duve",             # 《从美的艺术到一般艺术》作者
    "妮科尔·福斯格伦": "Nicole Forsgren",           # Accelerate 作者
    "劳伦斯·艾莉森": "Laurence Alison",
    "艾米丽·艾莉森": "Emily Alison",                # Rapport 合著者
    "埃米利·罗萨莱斯": "Emili Rosales",             # 《看不见的城市》作者
    "大卫·崔沛特": "David Trippett",
    "杰里米·阿罗尔斯": "Jeremy Aroles",
    "格雷格·D.贾克布": "Gregg D. Jacobs",           # Say Good Night to Insomnia
    "理查德·沃尔特": "Richard Walter",              # UCLA 编剧
    "路·贝·德·格瓦拉": "Luis Vélez de Guevara",     # 《瘸腿魔鬼》作者
    "阿德里安·德姆莱特纳": "Adrian Demleitner",      # 《数字人文研究》2025
    "阿琼·萨巴瓦尔": "Arjun Sabharwal",             # 《数字人文研究》2022
    "弗雷格": "Gottlob Frege", "鲁滨逊": "David Robinson",
    "施拉姆": "Wilbur Schramm",
    "彼得森": "Theodore Peterson",                  # 《传媒的四种理论》合著者
    "皮埃尔·布吕奈尔": "Pierre Brunel", "利昂·庞帕": "Leon Pompa",
    "弗里德里希·尼采": "Friedrich Nietzsche",
    "让-保罗·萨特": "Jean-Paul Sartre", "居伊·德波": "Guy Debord",
    "亨利·列斐伏尔": "Henri Lefebvre", "皮埃尔·布尔迪厄": "Pierre Bourdieu",
    "安东尼奥·葛兰西": "Antonio Gramsci", "爱德华·萨义德": "Edward Said",
    "苏珊·桑塔格": "Susan Sontag", "汉娜·阿伦特": "Hannah Arendt",
    "瓦尔特·本雅明": "Walter Benjamin", "西奥多·阿多诺": "Theodor W. Adorno",
}
CURATED_ALL: dict[str, str] = {**CURATED, **CURATED_FULL}

# 一条元素里塞了两个人：直接给整串的拆分结果
MULTI: dict[str, list[str]] = {
    "[美]欧文·M·柯匹 卡尔·柯恩": ["Irving M. Copi", "Carl Cohen"],
    "欧文·M·柯匹 卡尔·柯恩": ["Irving M. Copi", "Carl Cohen"],
}

# 明确保留的中文/日文/机构名：既不做译名转换，也不因「同行有拉丁作者」被删
KEEP_CJK = {"滨下武志", "牧尾晴喜", "永原康史", "凯丽比努尔·色提尼牙孜",
            "奥斯特洛夫斯基", "东之海神·西之沧海",
            "芸術アルゴリズム研究会·多摩美術大学", "扬布里柯"}
LEAD_TAG = re.compile(r"^\s*(?:[\[【〔（(]\s*[^\]】〕）)]{1,6}\s*[\]】〕）)])\s*")
TAIL_ROLE = re.compile(
    r"\s*(?:et al\.?|等编著|等编译|等译|等|编著|编译|著|译|编|校)\s*$")


# ------------------------------------------------------------------ 工具
def norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").strip()
    s = NATION_TAIL.sub("", s)
    if "," in s and is_latinish(s.split(",", 1)[0].strip()):
        a, b = s.split(",", 1)
        s = f"{b.strip()} {a.strip()}"
    t = re.sub(r"[\s.\-'’·．、，,]", "", s.lower())
    return "".join(c for c in unicodedata.normalize(
        "NFD", t) if not unicodedata.combining(c))


def is_latinish(s: str) -> bool:
    """按字符类别判拉丁（Ž/ž/ć/č 不在 LATINISH 的 U+00xx 区间里，别用它）。"""
    s = (s or "").strip()
    if not s or not s[0].isalpha():
        return False
    if KANA.search(s) or HANGUL.search(s):
        return False
    if any("一" <= c <= "鿿" for c in s):
        return False
    return all(c.isalpha() or c in " .'-," for c in s)


def looks_latin_person(s: str) -> bool:
    s = (s or "").strip()
    if not s or not is_latinish(s) or re.search(r"\d", s):
        return False
    return 2 <= len(s.replace(",", " ").split()) <= 6


def latin_names(names: list[str]) -> list[str]:
    out = []
    for n in names:
        n = (n or "").strip()
        if not n or n.lower() in JUNK:
            continue
        if "," in n and is_latinish(n.split(",", 1)[0].strip()):
            a, b = n.split(",", 1)
            n = f"{b.strip()} {a.strip()}"
        if looks_latin_person(n):
            out.append(n)
    return list(dict.fromkeys(out))


def filename_parts(path: str) -> list[str]:
    stem = os.path.basename(path).rsplit(".", 1)[0]
    return [p.strip() for p in stem.split(" - ") if p.strip()] if " - " in stem else []


def folder_author(path: str) -> str:
    seg = path.split("/")
    if len(seg) >= 3 and seg[0] in ("OneDrive", "Supplementary Books", "Authors"):
        return seg[1]
    return ""


def calibre_id(path: str) -> int | None:
    """取**最后一个** `(N)`：`(7)` 是卷号，取第一个会拿错书。"""
    ms = re.findall(r"\((\d+)\)", path)
    return int(ms[-1]) if ms else None


def is_junk(s: str) -> bool:
    return NATION_TAIL.sub("", s).strip().lower() in JUNK


# ------------------------------------------------------------------ 主流程
async def build_plan() -> tuple[list[dict], dict]:
    conn = await C.get_conn()
    try:
        rows = await conn.fetch(
            "SELECT id, path, coalesce(title,'') AS title, authors "
            "FROM file_metadata WHERE authors IS NOT NULL AND cardinality(authors)>0")
    finally:
        await conn.close()
    log.info("rows=%d", len(rows))

    cal = {int(k): v for k, v in json.load(open(
        "batch_operation/reports/calibre_authors.json", encoding="utf-8")).items()}
    z_items = json.load(open("batch_operation/reports/zotero_items.json",
                             encoding="utf-8"))
    z_exact: dict[str, list[str]] = {}
    for it in z_items:
        names = []
        for c in it.get("creators") or []:
            f = (c.get("firstName") or "").strip()
            l = (c.get("lastName") or "").strip()
            names.append(f"{f} {l}".strip() if f and l else (l or f))
        names = latin_names(names)
        k = re.sub(r"[\s\W_]+", "", unicodedata.normalize(
            "NFKC", it.get("title", "")).lower())
        if k and names and k not in z_exact:
            z_exact[k] = names
    log.info("calibre=%d zotero=%d", len(cal), len(z_exact))

    plan: list[dict] = []
    ev: Counter = Counter()

    for r in rows:
        path, title = r["path"], r["title"]
        authors = [a.strip() for a in (r["authors"] or []) if a and a.strip()]
        cjk = [a for a in authors if CJK_PURE.match(NATION_TAIL.sub("", a).strip())
               and not is_junk(a)]
        junky = [a for a in authors if is_junk(a)]
        new, tier, why = list(authors), "-", ""

        # 作者目录写的是中文 ⇒ 判定为中国作者，按约定保持中文名（译名不动）
        dir_raw = folder_author(path)
        dir_is_cjk = bool(dir_raw) and bool(
            CJK_PURE.match(NATION_TAIL.sub("", dir_raw).strip()))

        # ---- M1 人工表（只换写法、不换人，语义无变化 → 最先） ----
        if tier == "-":
            conv, hit = [], []
            for a in authors:
                if a.strip() in MULTI:                # 一条元素塞两个人
                    conv.extend(MULTI[a.strip()])
                    hit.append(a.strip())
                    continue
                base = NATION_TAIL.sub("", a).strip()
                if base in JP_NAMES:
                    conv.append(JP_NAMES[base])       # 日韩统一简体中文
                    hit.append(base)
                    continue
                # 清洗：去开头国别标签、去结尾角色词（等编著/编译/著/译…）
                core = TAIL_ROLE.sub("", LEAD_TAG.sub("", base)).strip()
                # 去掉裹在中间的角色尾巴（'约翰娜· 德鲁克 朱子彤／译' 这类不入表）
                tgt = CURATED_ALL.get(core) or CURATED_ALL.get(base)
                if tgt and tgt != base:
                    conv.append(tgt)              # 转成原名后不再保留 (国籍) 后缀
                    hit.append(core or base)
                else:
                    conv.append(a)
            if hit:
                new, tier, why = conv, "M1", "人工表/日韩简体 " + ",".join(hit)
                ev["M1"] += 1

        # ---- C1 Calibre 拉丁著者（换「人」：译者→著者，需硬证据） ----
        if tier == "-" and not dir_is_cjk:
            cid = calibre_id(path)
            cal_names = cal.get(cid, []) if cid else []
            cal_latin = latin_names(cal_names)
            if cal_latin and cjk:
                # 行内的中文名必须也在 Calibre 署名里（同一本书的署名，不是拿错书）
                if all(any(norm_key(a) == norm_key(c) for c in cal_names)
                       for a in cjk):
                    new = cal_latin
                    tier, why = "C1", f"Calibre book {cid} 著者={cal_latin}"
                    ev["C1"] += 1

        # ---- D1 目录名 / 文件名作者段（行内已有一个拉丁作者时不碰：
        #      多出的中文可能是共同著者的译名，替换会丢人） ----
        has_latin = any(looks_latin_person(NATION_TAIL.sub("", a))
                        for a in authors)
        if tier == "-" and not dir_is_cjk and not has_latin:
            parts = filename_parts(path)
            cand = ""
            if len(parts) >= 2:
                head, tailp = parts[0], parts[-1]
                h_l = latin_names([head])
                t_l = latin_names([tailp])
                if t_l and not h_l:
                    cand = t_l[0]
                elif h_l and not t_l and len(parts) == 2:
                    cand = h_l[0]          # Authors/ 树 `<作者> - <题名>` 写法
            d_l = latin_names([dir_raw])
            if not cand and d_l:
                if not any(latin_names([p]) and p != dir_raw for p in parts):
                    cand = d_l[0]
            if cand and cjk and norm_key(cand) not in {norm_key(a) for a in cjk}:
                new = [cand]
                tier, why = "D1", f"目录/文件名作者='{cand}'"
                ev["D1"] += 1

        # ---- Z1 外国标记 + Zotero 精确命中单著者 ----
        if tier == "-" and cjk and len(cjk) == 1 and not has_latin:
            foreign = bool(NATION_MARK.search(title) or NATION_MARK.search(path)
                           or any(NATION_TAIL.search(a) for a in authors))
            k = re.sub(r"[\s\W_]+", "", unicodedata.normalize("NFKC", title).lower())
            zn = z_exact.get(k, [])
            if foreign and len(zn) == 1:
                new = [zn[0]]
                tier, why = "Z1", f"Zotero精确 {cjk[0]}→{zn[0]}"
                ev["Z1"] += 1

        # ---- T5cn：中文作者被写成「中文名_拼音名」两遍（Zotero/<中文>_<拼音>_<年>_…）
        #      → 保留中文名、删拼音（中文作者按 姓+名 约定写作） ----
        if tier == "-" and len(cjk) == 1 and path.startswith("Zotero/"):
            lat_all = [x for x in new if looks_latin_person(NATION_TAIL.sub("", x))]
            stem = os.path.basename(path).rsplit(".", 1)[0]
            segs0 = [x.strip() for x in stem.split("_") if x.strip()]
            if len(lat_all) == 1 and len(segs0) >= 2 \
                    and segs0[0] == cjk[0] and segs0[1] == lat_all[0]:
                new, tier = [cjk[0]], "T5cn"
                why = "中文名_拼音名重复 → 保留中文（中文作者 姓+名）"
                ev["T5cn"] += 1

        # ---- T5drop：同行已有拉丁作者 → 纯中文元素是译者/编者，删掉 ----
        # 用户 2026-10-11 指令：「删掉中文名」。KEEP_CJK 例外。
        if any(looks_latin_person(NATION_TAIL.sub("", x)) for x in new):
            keep = [x for x in new if x in KEEP_CJK
                    or not (CJK_PURE.match(x.strip()) and x.strip() not in KEEP_CJK)]
            if keep != new and keep:
                dropped = [x for x in new if x not in keep]
                if tier == "-":
                    tier, why = "T5drop", ""
                why = (why + "; " if why else "") + "删中文译者/编者 " + ",".join(dropped)
                new = keep
                ev["T5drop"] += 1

        # ---- X1 未知→补名（单独档，默认不写） ----
        if tier == "-" and junky:
            cid = calibre_id(path)
            cal_latin = latin_names(cal.get(cid, [])) if cid else []
            d_l = latin_names([folder_author(path)])
            cand = cal_latin[0] if cal_latin else (d_l[0] if d_l else "")
            if cand:
                new = [x for x in authors if not is_junk(x)] + [cand]
                tier, why = "X1", f"未知→{cand}"
                ev["X1"] += 1

        new = list(dict.fromkeys(x for x in new if x))
        plan.append({"id": str(r["id"]), "path": path, "old_title": title,
                     "old_authors": authors, "new_authors": new,
                     "tier": tier, "why": why, "change": new != authors})

    # ---- U1 异写合并 ----
    clusters: dict[str, Counter] = defaultdict(Counter)
    for p in plan:
        for a in p["new_authors"]:
            if not is_junk(a):
                clusters[norm_key(a)][a] += 1
    canon: dict[str, str] = {}
    for key, cnt in clusters.items():
        forms = list(cnt)
        if len(forms) < 2 or len(forms) > 5:
            continue
        lat = [f for f in forms if looks_latin_person(NATION_TAIL.sub("", f))]
        cjkf = [f for f in forms if f not in lat]
        def lat_pref(x: str):
            has_accent = int(any(ord(c) > 127 for c in x))
            return (-has_accent, -(x.count(".") + x.count("-")),
                    int(x.isupper()), -cnt[x])
        if lat and cjkf:
            pick = sorted(lat, key=lat_pref)[0]
        elif len(lat) >= 2:
            # 全拉丁：优先带重音/缩写点/连字符的完整形（'G. E. M. Anscombe' > 'GEM Anscombe'）
            pick = sorted(lat, key=lat_pref)[0]
        elif len(lat) == 1:
            pick = lat[0]
        else:
            keys = {re.sub(r"[\s·]", "", unicodedata.normalize("NFKC", f))
                    for f in forms}
            if len(keys) != 1:
                continue
            # 全中文：优先带国籍后缀的完整形，再看频次
            pick = sorted(forms, key=lambda x: (-bool(NATION_TAIL.search(x)),
                                                 -cnt[x], -len(x)))[0]
        for f in forms:
            if f != pick:
                canon[f] = pick

    merged_rows = 0
    for p in plan:
        conv = list(dict.fromkeys(canon.get(a, a) for a in p["new_authors"]))
        if conv != p["new_authors"]:
            moved = sorted({a for a in p["new_authors"] if canon.get(a, a) != a})
            p["why"] = (p["why"] + "; " if p["why"] else "") + \
                "合并异写:" + ",".join(moved[:3])
            p["new_authors"] = conv
            p["change"] = conv != p["old_authors"]
            if p["tier"] == "-":
                p["tier"] = "U1"
                ev["U1"] += 1
            merged_rows += 1
        if p["new_authors"] != p["old_authors"] and p["tier"] == "-":
            p["tier"] = "U1"
            p["change"] = True

    log.info("change=%d ev=%s merged_rows=%d canon_pairs=%d",
             sum(1 for p in plan if p["change"]), dict(ev), merged_rows, len(canon))
    return plan, canon


async def run(apply: bool = False) -> dict:
    plan, canon = await build_plan()
    changes = [p for p in plan if p["change"]]
    by_tier = Counter(p["tier"] for p in changes)
    summary = {"applied": apply, "rows": len(plan), "to_update": len(changes),
               "by_tier": dict(by_tier), "canonical_pairs": len(canon)}
    os.makedirs("batch_operation/reports", exist_ok=True)
    with open("batch_operation/reports/authors_unify_plan.json", "w",
              encoding="utf-8") as f:
        json.dump({"summary": summary, "canonical": canon, "plans": plan},
                  f, ensure_ascii=False, indent=1)
    with open("batch_operation/reports/authors_unify_plan.tsv", "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["tier", "path", "old_authors", "new_authors", "why"])
        for p in changes:
            w.writerow([p["tier"], p["path"],
                        "{" + ",".join(p["old_authors"]) + "}",
                        "{" + ",".join(p["new_authors"]) + "}", p["why"]])

    for p in changes:
        p["new_authors"] = list(dict.fromkeys(
            LATIN_CANON.get(a, a) for a in p["new_authors"]))
    todo = changes
    summary["x1_deferred"] = 0
    if apply and todo:
        # 备份名带时分秒：同一天重复 apply 不许互相覆盖（实测吃过一次亏——
        # 第二次 apply 把 611 行的 pre-image 覆盖成了 117 行）
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        bak = f"batch_operation/reports/authors_unify_backup_{stamp}.tsv"
        with open(bak, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["id", "path", "authors"])
            for p in todo:
                w.writerow([p["id"], p["path"],
                            "{" + ",".join(p["old_authors"]) + "}"])
        conn = await C.get_conn()
        try:
            # 护栏用**数组参数**比对：`authors::text` 会给含空格的元素加引号
            # （`{"Theodor W Adorno"}`），拿手拼的 `{Theodor W Adorno}` 去比必然失配，
            # 结果是整批静默跳过、日志还报“updated=N”。
            await conn.executemany(
                "UPDATE file_metadata SET authors = $2::text[] WHERE id = $1 "
                "AND authors = $3::text[]",
                [(p["id"], p["new_authors"], p["old_authors"]) for p in todo])
        finally:
            await conn.close()
        summary["backup"] = bak
        summary["applied_rows"] = len(todo)
        log.info("[apply] updated=%d backup=%s", len(todo), bak)
    log.info("[summary] %s", json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--tier", default="")
    args = ap.parse_args()
    summary = asyncio.run(run(apply=args.apply))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.show:
        d = json.load(open("batch_operation/reports/authors_unify_plan.json",
                           encoding="utf-8"))
        n = 0
        for p in d["plans"]:
            if p["change"] and (not args.tier or p["tier"] == args.tier):
                if n >= args.show:
                    break
                n += 1
                print(f"[{p['tier']}] {p['path'][:78]}")
                print(f"     {p['old_authors']} -> {p['new_authors']}")
                print(f"     {p['why'][:120]}")


if __name__ == "__main__":
    main()
