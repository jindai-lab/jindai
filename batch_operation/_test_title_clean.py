import importlib.util
src = open("/app/batch_operation/authors_tree_clean.py", encoding="utf-8").read()
ns = {}; exec(src.split("async def main")[0].replace("import common as C", "C = None"), ns)
CASES = [
    ("Hardt, Michael", "Labor of Dionysus_ A Critique of the State-Form _Michael Hardt & Antonio Negri"),
    ("Heller, Agnes", "Aesthetics_and_Modernity__Essays_by_Agnes_Heller_Heller,"),
    ("Ilee DeSoto", "The Art of Algorithms_ Exploring AI in Creative Industries -- Ilee DeSoto -- 2024 -- Anna\u2019s Archive"),
    ("Martin, Reinhold", "The Organizational Complex_ Architecture, Media, and -- Reinhold Martin -- 2003"),
    ("Martin, Reinhold", "Utopia\u2019s Ghost_ Architecture and Postmodernism, Again -- Reinhold Martin -- 2010"),
    ("Noe, Alva", "The Entanglement_ How Art and Philosophy Make Us What We Are_Alva No\u00eb"),
    ("Levinson, Paul", "Digital McLuhan_ A Guide to the Information Millennium -- Paul Levinson"),
    ("Culler, Jonathan", "Culler, Jonathan D."),
    ("Deren, Maya", "Maya Deren and the American Avant-Garde"),
    ("Badiou, Alain", "In Praise of Love_Alain Badiou, Nicolas Truong, Translated by Peter Bush"),
    ("Arendt, Hannah", "Arendt Hannah The Origins of Totalitarianism 1979"),
]
for d, t in CASES:
    da = ns["dir_author"](d)
    print(f"  {d:17s} {t[:52]:52s} -> {ns['clean_title'](t, t, da, [da])[0][:48]}")
