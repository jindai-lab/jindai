# Supplementary Books 根级 42 个 PDF 的推定作者（证据）

2026-10-10 用户指令：「Supplementary Books/ 根级 42 个根据文本内容或搜索来推测作者并挂到书库--作者下」。
方法：pymupdf 提取前两页文本（`_probe_supp_root_titles.py`，中文扫描件无文本层则回退到文件名书目信息），
英文本再取 p1–p16（`_probe_deep.py`），存疑的用网络查证。结果写入
`backfill_folder_datasets.py::SUPP_ROOT_AUTHORS`（规则的一处真源）。

## 一、题名页文本直接给出作者（英文本）

| 文件 | 推定作者 | 依据 |
|---|---|---|
| Chinese_Researches.pdf | Alexander Wylie | p2 题名页「Chinese researches — Alexander Wylie, Henri Cordier」 |
| Conic_Sections.pdf | William Whewell | p6/p8「CONIC SECTIONS … BY WILLIAM WHEWELL, D.D., Master of Trinity College, Cambridge」 |
| Memorial Wylie.pdf | Alexander Wylie | 实为 *Memorials of Protestant Missionaries to the Chinese*, Shanghai: A. P. Mission Press, 1867（Wylie 编），文件名即「Memorial Wylie」 |
| Shanghai Country Walks.pdf | E. S. Wilkinson | p3 题名页「SHANGHAI COUNTRY WALKS … by E. S. WILKINSON, Shanghai: North-China Daily News」 |
| The_treaty_ports_of_China_and_Japan_a_gu.pdf | N. B. Dennys | 1867 London: Trübner；「Wm. F. Mayers, N. B. Dennys, Ch. King — compiled and edited by N. B. Dennys」（Cambridge/HathiTrust/BNASIE 书目一致） |
| hangchow itineraries.pdf | Robert F. Fitch | p1「HANGCHOW ITINERARIES By Robert F. Fitch」 |
| A chinese child's day… | Anice Terhune | 文件名「By Anice Terhune」（插画 Albertine Randall Wheelan，非作者） |
| Illustrations of China and its people… | John Thomson | 文件名「By John Thomson」 |

## 二、文件名书目信息（中文古籍，扫描件无文本层）

| 文件 | 推定作者 | 依据 |
|---|---|---|
| 万寿盛典初集… | 王原祁 | 「王原祁等纂修」 |
| 三国志… | 陈寿 | 「陈寿撰．裴松之注」 |
| 六朝事迹编类… | 张敦颐 | 「宋张敦颐撰」 |
| 吴郡名贤图传赞… | 顾沅 | 「清顾沅撰」 |
| 大义觉迷录… | 雍正 | 「清．雍正编」 |
| 奇姓通… | 夏树芳 | 「明夏树芳辑」 |
| 支那通史… | 那珂通世 | 「那珂通世编」 |
| 无双谱… | 金古良 | 「清．金古良编绘」 |
| 明状元图考… | 顾祖训 | 「明顾祖训辑」 |
| 最新中国历史教科书… | 姚祖义 | 「姚祖义编」 |
| 清异录… | 陶谷 | 「宋陶谷撰」 |
| 皇清职贡图… | 傅恒 | 「清傅恒等纂」 |
| 诸史将略… | 刘畿 | 「明刘畿纂辑」 |
| 诸葛忠武侯年谱… | 杨希闵 | 「杨希闵编」 |
| 镌重订补注历朝捷录… | 顾充 | 「顾充辑著」 |
| 回忆录_罗宗洛.pdf | 罗宗洛 | p1「罗宗洛遗著」 |
| 浙江图书馆丛书.pdf | 浙江图书馆 | 丛书编辑者即机构 |
| 元和郡县图志1–10.pdf | 李吉甫 | 唐李吉甫撰，通行本；10 个分卷文件同一作者 |

## 三、网络查证

| 文件 | 推定作者 | 依据 |
|---|---|---|
| 人民周刊_1926年2月——1927年4月.pdf | 张太雷 | 中共广东区委机关报，1926-02-07 广州创刊，张太雷创办并首任主编，共 50 期（marxists.org 目录、南方+、中国共青团新闻网） |
| 北京各行祖师调查记略.pdf | 崇璋 | 即《北平各行祖师调查记略》，旗人崇璋著（界面新闻引书、中国民俗学网述刘佳崇璋该调查） |
| The_treaty_ports… | N. B. Dennys | 同上 |
| 古先君臣图鉴.附小传古赞.明代.pdf | 佚名 | 明万历益藩藩府刻本，编者不详（书格 shuge.org、搜狐版本学文章）→ 按馆例归「佚名」 |
| 吕氏春秋.漢.高誘註.pdf | 吕不韦 | 战国吕不韦召集门客撰，高诱为注者 |
| 战国策.12策.闵齐伋裁注… | 刘向 | 西汉刘向校订编定；闵齐伋为评注者、乌程闵齐伋刊为版本 |
| 八旬万寿盛典图说… | 乾隆 | 清高宗（乾隆）敕撰，乾隆五十七年武英殿套印本 |
| 光绪元年致日本开拓判官照会册… | 直隶津海关道 | 稿本发文机关即著者（清代公牍通例） |

存疑可改：`崇璋`（亦有文献作「刘佳崇璋」）、`乾隆`（或作「清高宗」）、`佚名`（古先君臣图鉴）。
