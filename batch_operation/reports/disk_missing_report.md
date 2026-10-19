# jindai 文库：记录在库但本地文件不在盘的清单

- 盘点范围：`file_metadata` 全库 274,000 行（逐行 `os.path.exists` 实测）
- 命中「本地路径但文件不在盘」：**1305** 行
- 其中父目录存在（文件被删/改名）：**311** 行；父目录也不存在（整棵目录树缺失）：**994** 行

## 按顶层目录分布

| 顶层目录 | 行数 | 该目录是否在盘 |
|---|---:|---|
| `Journals/` | 577 | 是 |
| `Authors/` | 391 | 是 |
| `Zotero/` | 229 | 是 |
| `OneDrive/` | 80 | 是 |
| `Series/` | 15 | 是 |
| `Albums/` | 12 | 是 |
| `Ernst Bloch/` | 1 | **否** |


## Journals/（577 行）

| # | 原路径 | 文件名 | 标题 | 作者 | 父目录在盘 |
|---:|---|---|---|---|---|
| 1 | `Journals/COMPUTER_GRAPHICS_AND_ART_Aug1977.pdf` | COMPUTER_GRAPHICS_AND_ART_Aug1977.pdf | COMPUTER_GRAPHICS_AND_ART_Aug1977 |  | 是 |
| 2 | `Journals/COMPUTER_GRAPHICS_AND_ART_Aug1978.pdf` | COMPUTER_GRAPHICS_AND_ART_Aug1978.pdf | COMPUTER_GRAPHICS_AND_ART_Aug1978 |  | 是 |
| 3 | `Journals/COMPUTER_GRAPHICS_AND_ART_Feb1977.pdf` | COMPUTER_GRAPHICS_AND_ART_Feb1977.pdf | COMPUTER_GRAPHICS_AND_ART_Feb1977 |  | 是 |
| 4 | `Journals/COMPUTER_GRAPHICS_AND_ART_Feb1978.pdf` | COMPUTER_GRAPHICS_AND_ART_Feb1978.pdf | COMPUTER_GRAPHICS_AND_ART_Feb1978 |  | 是 |
| 5 | `Journals/COMPUTER_GRAPHICS_AND_ART_May1976.pdf` | COMPUTER_GRAPHICS_AND_ART_May1976.pdf | COMPUTER_GRAPHICS_AND_ART_May1976 |  | 是 |
| 6 | `Journals/COMPUTER_GRAPHICS_AND_ART_May1977.pdf` | COMPUTER_GRAPHICS_AND_ART_May1977.pdf | COMPUTER_GRAPHICS_AND_ART_May1977 |  | 是 |
| 7 | `Journals/COMPUTER_GRAPHICS_AND_ART_May1978.pdf` | COMPUTER_GRAPHICS_AND_ART_May1978.pdf | COMPUTER_GRAPHICS_AND_ART_May1978 |  | 是 |
| 8 | `Journals/COMPUTER_GRAPHICS_AND_ART_Nov1976.pdf` | COMPUTER_GRAPHICS_AND_ART_Nov1976.pdf | COMPUTER_GRAPHICS_AND_ART_Nov1976 |  | 是 |
| 9 | `Journals/COMPUTER_GRAPHICS_AND_ART_Nov1977.pdf` | COMPUTER_GRAPHICS_AND_ART_Nov1977.pdf | COMPUTER_GRAPHICS_AND_ART_Nov1977 |  | 是 |
| 10 | `Journals/ChinaJournal/oregondigital_df72cv681` | oregondigital_df72cv681 |  |  | 否 |
| 11 | `Journals/ChinaJournal/oregondigital_df72cz83r` | oregondigital_df72cz83r |  |  | 否 |
| 12 | `Journals/ChinaJournal/oregondigital_df72d1202` | oregondigital_df72d1202 |  |  | 否 |
| 13 | `Journals/ChinaJournal/oregondigital_df72d1245` | oregondigital_df72d1245 |  |  | 否 |
| 14 | `Journals/ChinaJournal/oregondigital_df72d132v` | oregondigital_df72d132v |  |  | 否 |
| 15 | `Journals/ChinaJournal/oregondigital_df72d174c` | oregondigital_df72d174c |  |  | 否 |
| 16 | `Journals/ChinaJournal/oregondigital_df72d312s` | oregondigital_df72d312s |  |  | 否 |
| 17 | `Journals/ChinaJournal/oregondigital_df72d319q` | oregondigital_df72d319q |  |  | 否 |
| 18 | `Journals/ChinaJournal/oregondigital_df72d3620` | oregondigital_df72d3620 |  |  | 否 |
| 19 | `Journals/ChinaJournal/oregondigital_df72d446t` | oregondigital_df72d446t |  |  | 否 |
| 20 | `Journals/ChinaJournal/oregondigital_df72d454h` | oregondigital_df72d454h |  |  | 否 |
| 21 | `Journals/ChinaJournal/oregondigital_df72d468v` | oregondigital_df72d468v |  |  | 否 |
| 22 | `Journals/ChinaJournal/oregondigital_df72d479c` | oregondigital_df72d479c |  |  | 否 |
| 23 | `Journals/ChinaJournal/oregondigital_df72d481d` | oregondigital_df72d481d |  |  | 否 |
| 24 | `Journals/ChinaJournal/oregondigital_df72d533z` | oregondigital_df72d533z |  |  | 否 |
| 25 | `Journals/ChinaJournal/oregondigital_df72d548k` | oregondigital_df72d548k |  |  | 否 |
| 26 | `Journals/ChinaJournal/oregondigital_df72d621c` | oregondigital_df72d621c |  |  | 否 |
| 27 | `Journals/ChinaJournal/oregondigital_df72d632w` | oregondigital_df72d632w |  |  | 否 |
| 28 | `Journals/ChinaJournal/oregondigital_df72d643d` | oregondigital_df72d643d |  |  | 否 |
| 29 | `Journals/ChinaJournal/oregondigital_df72d6492` | oregondigital_df72d6492 |  |  | 否 |
| 30 | `Journals/ChinaJournal/oregondigital_df72d672v` | oregondigital_df72d672v |  |  | 否 |
| 31 | `Journals/ChinaJournal/oregondigital_df72d694w` | oregondigital_df72d694w |  |  | 否 |
| 32 | `Journals/ChinaJournal/oregondigital_df72d7391` | oregondigital_df72d7391 |  |  | 否 |
| 33 | `Journals/ChinaJournal/oregondigital_df72d746f` | oregondigital_df72d746f |  |  | 否 |
| 34 | `Journals/ChinaJournal/oregondigital_df72d830r` | oregondigital_df72d830r |  |  | 否 |
| 35 | `Journals/ChinaJournal/oregondigital_df72d854b` | oregondigital_df72d854b |  |  | 否 |
| 36 | `Journals/ChinaJournal/oregondigital_df72d8796` | oregondigital_df72d8796 |  |  | 否 |
| 37 | `Journals/ChinaJournal/oregondigital_df72d923j` | oregondigital_df72d923j |  |  | 否 |
| 38 | `Journals/ChinaJournal/oregondigital_df72d9317` | oregondigital_df72d9317 |  |  | 否 |
| 39 | `Journals/ChinaJournal/oregondigital_df72d973r` | oregondigital_df72d973r |  |  | 否 |
| 40 | `Journals/ChinaJournal/oregondigital_df72d9805` | oregondigital_df72d9805 |  |  | 否 |
| 41 | `Journals/ChinaJournal/oregondigital_df72db10p` | oregondigital_df72db10p |  |  | 否 |
| 42 | `Journals/ChinaJournal/oregondigital_df72db22g` | oregondigital_df72db22g |  |  | 否 |
| 43 | `Journals/ChinaJournal/oregondigital_df72dd66z` | oregondigital_df72dd66z |  |  | 否 |
| 44 | `Journals/ChinaJournal/oregondigital_df72dd791` | oregondigital_df72dd791 |  |  | 否 |
| 45 | `Journals/ChinaJournal/oregondigital_df72df51j` | oregondigital_df72df51j |  |  | 否 |
| 46 | `Journals/ChinaJournal/oregondigital_df72df711` | oregondigital_df72df711 |  |  | 否 |
| 47 | `Journals/ChinaJournal/oregondigital_df72dg44b` | oregondigital_df72dg44b |  |  | 否 |
| 48 | `Journals/ChinaJournal/oregondigital_df72dh62g` | oregondigital_df72dh62g |  |  | 否 |
| 49 | `Journals/ChinaJournal/oregondigital_df72dh659` | oregondigital_df72dh659 |  |  | 否 |
| 50 | `Journals/ChinaJournal/oregondigital_df72dj893` | oregondigital_df72dj893 |  |  | 否 |
| 51 | `Journals/ChinaJournal/oregondigital_df72dk03q` | oregondigital_df72dk03q |  |  | 否 |
| 52 | `Journals/ChinaJournal/oregondigital_df72dm224` | oregondigital_df72dm224 |  |  | 否 |
| 53 | `Journals/ChinaJournal/oregondigital_df72dq19f` | oregondigital_df72dq19f |  |  | 否 |
| 54 | `Journals/ChinaJournal/oregondigital_df72dq320` | oregondigital_df72dq320 |  |  | 否 |
| 55 | `Journals/ChinaJournal/oregondigital_df72dq48w` | oregondigital_df72dq48w |  |  | 否 |
| 56 | `Journals/ChinaJournal/oregondigital_df72dr15j` | oregondigital_df72dr15j |  |  | 否 |
| 57 | `Journals/ChinaJournal/oregondigital_df72dr56s` | oregondigital_df72dr56s |  |  | 否 |
| 58 | `Journals/ChinaJournal/oregondigital_df72ds293` | oregondigital_df72ds293 |  |  | 否 |
| 59 | `Journals/ChinaJournal/oregondigital_df72ds692` | oregondigital_df72ds692 |  |  | 否 |
| 60 | `Journals/ChinaJournal/oregondigital_df72ds85f` | oregondigital_df72ds85f |  |  | 否 |
| 61 | `Journals/ChinaJournal/oregondigital_df72dv44m` | oregondigital_df72dv44m |  |  | 否 |
| 62 | `Journals/ChinaJournal/oregondigital_df72dv88p` | oregondigital_df72dv88p |  |  | 否 |
| 63 | `Journals/ChinaJournal/oregondigital_df72dw59f` | oregondigital_df72dw59f |  |  | 否 |
| 64 | `Journals/ChinaJournal/oregondigital_df72dw78n` | oregondigital_df72dw78n |  |  | 否 |
| 65 | `Journals/ChinaJournal/oregondigital_df72dw88w` | oregondigital_df72dw88w |  |  | 否 |
| 66 | `Journals/ChinaJournal/oregondigital_df72dx505` | oregondigital_df72dx505 |  |  | 否 |
| 67 | `Journals/ChinaJournal/oregondigital_df72dx56t` | oregondigital_df72dx56t |  |  | 否 |
| 68 | `Journals/ChinaJournal/oregondigital_df72dx58c` | oregondigital_df72dx58c |  |  | 否 |
| 69 | `Journals/ChinaJournal/oregondigital_df72dz52x` | oregondigital_df72dz52x |  |  | 否 |
| 70 | `Journals/ChinaJournal/oregondigital_df72f000j` | oregondigital_df72f000j |  |  | 否 |
| 71 | `Journals/ChinaJournal/oregondigital_df72f029h` | oregondigital_df72f029h |  |  | 否 |
| 72 | `Journals/ChinaJournal/oregondigital_df72f0554` | oregondigital_df72f0554 |  |  | 否 |
| 73 | `Journals/ChinaJournal/oregondigital_df72f085v` | oregondigital_df72f085v |  |  | 否 |
| 74 | `Journals/ChinaJournal/oregondigital_df72f124b` | oregondigital_df72f124b |  |  | 否 |
| 75 | `Journals/ChinaJournal/oregondigital_df72f128f` | oregondigital_df72f128f |  |  | 否 |
| 76 | `Journals/ChinaJournal/oregondigital_df72k0296` | oregondigital_df72k0296 |  |  | 否 |
| 77 | `Journals/ChinaJournal/oregondigital_df72k0385` | oregondigital_df72k0385 |  |  | 否 |
| 78 | `Journals/ChinaJournal/oregondigital_df72k067m` | oregondigital_df72k067m |  |  | 否 |
| 79 | `Journals/ChinaJournal/oregondigital_df72k1683` | oregondigital_df72k1683 |  |  | 否 |
| 80 | `Journals/ChinaJournal/oregondigital_df72k2069` | oregondigital_df72k2069 |  |  | 否 |
| 81 | `Journals/ChinaJournal/oregondigital_df72k2582` | oregondigital_df72k2582 |  |  | 否 |
| 82 | `Journals/ChinaJournal/oregondigital_df72k3481` | oregondigital_df72k3481 |  |  | 否 |
| 83 | `Journals/ChinaJournal/oregondigital_df72k3502` | oregondigital_df72k3502 |  |  | 否 |
| 84 | `Journals/ChinaJournal/oregondigital_df72k380s` | oregondigital_df72k380s |  |  | 否 |
| 85 | `Journals/ChinaJournal/oregondigital_df72rr29m` | oregondigital_df72rr29m |  |  | 否 |
| 86 | `Journals/ChinaJournal/oregondigital_df72rr40m` | oregondigital_df72rr40m |  |  | 否 |
| 87 | `Journals/ChinaJournal/oregondigital_df72rr92c` | oregondigital_df72rr92c |  |  | 否 |
| 88 | `Journals/ChinaJournal/oregondigital_df72rs103` | oregondigital_df72rs103 |  |  | 否 |
| 89 | `Journals/ChinaJournal/oregondigital_df72rs51b` | oregondigital_df72rs51b |  |  | 否 |
| 90 | `Journals/ChinaJournal/oregondigital_df72rs66z` | oregondigital_df72rs66z |  |  | 否 |
| 91 | `Journals/ChinaJournal/oregondigital_df72rt05f` | oregondigital_df72rt05f |  |  | 否 |
| 92 | `Journals/ChinaJournal/oregondigital_df72rt380` | oregondigital_df72rt380 |  |  | 否 |
| 93 | `Journals/ChinaJournal/oregondigital_df72rv46w` | oregondigital_df72rv46w |  |  | 否 |
| 94 | `Journals/ChinaJournal/oregondigital_df72rv59z` | oregondigital_df72rv59z |  |  | 否 |
| 95 | `Journals/ChinaJournal/oregondigital_df72rv849` | oregondigital_df72rv849 |  |  | 否 |
| 96 | `Journals/ChinaJournal/oregondigital_df72rw38d` | oregondigital_df72rw38d |  |  | 否 |
| 97 | `Journals/ChinaJournal/oregondigital_df72rw71f` | oregondigital_df72rw71f |  |  | 否 |
| 98 | `Journals/ChinaJournal/oregondigital_df72rx78k` | oregondigital_df72rx78k |  |  | 否 |
| 99 | `Journals/Whole Earth Catalog/WEC 1969-1.pdf` | WEC 1969-1.pdf | WEC 1969-1 |  | 是 |
| 100 | `Journals/Whole Earth Catalog/WEC 1971-99.pdf` | WEC 1971-99.pdf | WEC 1971-99 |  | 是 |
| 101 | `Journals/史料/东方杂志/东方杂志1910年第7卷第1期.djvu` | 东方杂志1910年第7卷第1期.djvu |  |  | 否 |
| 102 | `Journals/史料/东方杂志/东方杂志1910年第7卷第2期.djvu` | 东方杂志1910年第7卷第2期.djvu |  |  | 否 |
| 103 | `Journals/史料/东方杂志/东方杂志1910年第7卷第3期.djvu` | 东方杂志1910年第7卷第3期.djvu |  |  | 否 |
| 104 | `Journals/史料/东方杂志/东方杂志1910年第7卷第4期.djvu` | 东方杂志1910年第7卷第4期.djvu |  |  | 否 |
| 105 | `Journals/史料/东方杂志/东方杂志1910年第7卷第5期.djvu` | 东方杂志1910年第7卷第5期.djvu |  |  | 否 |
| 106 | `Journals/史料/东方杂志/东方杂志1910年第7卷第6期.djvu` | 东方杂志1910年第7卷第6期.djvu |  |  | 否 |
| 107 | `Journals/史料/东方杂志/东方杂志1911年2月第8卷第1期.djvu` | 东方杂志1911年2月第8卷第1期.djvu |  |  | 否 |
| 108 | `Journals/史料/东方杂志/东方杂志1911年3月第8卷第2期.djvu` | 东方杂志1911年3月第8卷第2期.djvu |  |  | 否 |
| 109 | `Journals/史料/东方杂志/东方杂志1911年4月第8卷第3期.djvu` | 东方杂志1911年4月第8卷第3期.djvu |  |  | 否 |
| 110 | `Journals/史料/东方杂志/东方杂志1911年6月第8卷第5期.djvu` | 东方杂志1911年6月第8卷第5期.djvu |  |  | 否 |
| 111 | `Journals/史料/东方杂志/东方杂志1911年6月第8卷第6期.djvu` | 东方杂志1911年6月第8卷第6期.djvu |  |  | 否 |
| 112 | `Journals/史料/东方杂志/东方杂志1911年7月第8卷第7期.djvu` | 东方杂志1911年7月第8卷第7期.djvu |  |  | 否 |
| 113 | `Journals/史料/东方杂志/东方杂志1911年8月第8卷第8期.djvu` | 东方杂志1911年8月第8卷第8期.djvu |  |  | 否 |
| 114 | `Journals/史料/东方杂志/东方杂志1911年9月第8卷第9期.djvu` | 东方杂志1911年9月第8卷第9期.djvu |  |  | 否 |
| 115 | `Journals/史料/东方杂志/东方杂志1912年07月第9卷第1期.djvu` | 东方杂志1912年07月第9卷第1期.djvu |  |  | 否 |
| 116 | `Journals/史料/东方杂志/东方杂志1912年09月第9卷第3期.djvu` | 东方杂志1912年09月第9卷第3期.djvu |  |  | 否 |
| 117 | `Journals/史料/东方杂志/东方杂志1912年10月第9卷第4期.djvu` | 东方杂志1912年10月第9卷第4期.djvu |  |  | 否 |
| 118 | `Journals/史料/东方杂志/东方杂志1912年11月第9卷第5期.djvu` | 东方杂志1912年11月第9卷第5期.djvu |  |  | 否 |
| 119 | `Journals/史料/东方杂志/东方杂志1912年12月第9卷第6期.djvu` | 东方杂志1912年12月第9卷第6期.djvu |  |  | 否 |
| 120 | `Journals/史料/东方杂志/东方杂志1913年07月第10卷第1期.djvu` | 东方杂志1913年07月第10卷第1期.djvu |  |  | 否 |
| 121 | `Journals/史料/东方杂志/东方杂志1913年08月第10卷第2期.djvu` | 东方杂志1913年08月第10卷第2期.djvu |  |  | 否 |
| 122 | `Journals/史料/东方杂志/东方杂志1913年09月第10卷第3期.djvu` | 东方杂志1913年09月第10卷第3期.djvu |  |  | 否 |
| 123 | `Journals/史料/东方杂志/东方杂志1913年10月第10卷第4期.djvu` | 东方杂志1913年10月第10卷第4期.djvu |  |  | 否 |
| 124 | `Journals/史料/东方杂志/东方杂志1913年11月第10卷第5期.djvu` | 东方杂志1913年11月第10卷第5期.djvu |  |  | 否 |
| 125 | `Journals/史料/东方杂志/东方杂志1913年12月第10卷第6期.djvu` | 东方杂志1913年12月第10卷第6期.djvu |  |  | 否 |
| 126 | `Journals/史料/东方杂志/东方杂志1914年07月第11卷第1期.djvu` | 东方杂志1914年07月第11卷第1期.djvu |  |  | 否 |
| 127 | `Journals/史料/东方杂志/东方杂志1914年08月第11卷第2期.djvu` | 东方杂志1914年08月第11卷第2期.djvu |  |  | 否 |
| 128 | `Journals/史料/东方杂志/东方杂志1914年09月第11卷第3期.djvu` | 东方杂志1914年09月第11卷第3期.djvu |  |  | 否 |
| 129 | `Journals/史料/东方杂志/东方杂志1914年10月第11卷第4期.djvu` | 东方杂志1914年10月第11卷第4期.djvu |  |  | 否 |
| 130 | `Journals/史料/东方杂志/东方杂志1914年11月第11卷第5期.djvu` | 东方杂志1914年11月第11卷第5期.djvu |  |  | 否 |
| 131 | `Journals/史料/东方杂志/东方杂志1914年12月第11卷第6期.djvu` | 东方杂志1914年12月第11卷第6期.djvu |  |  | 否 |
| 132 | `Journals/史料/东方杂志/东方杂志1915年第12卷第10期.djvu` | 东方杂志1915年第12卷第10期.djvu |  |  | 否 |
| 133 | `Journals/史料/东方杂志/东方杂志1915年第12卷第11期.djvu` | 东方杂志1915年第12卷第11期.djvu |  |  | 否 |
| 134 | `Journals/史料/东方杂志/东方杂志1915年第12卷第12期.djvu` | 东方杂志1915年第12卷第12期.djvu |  |  | 否 |
| 135 | `Journals/史料/东方杂志/东方杂志1915年第12卷第1期.djvu` | 东方杂志1915年第12卷第1期.djvu |  |  | 否 |
| 136 | `Journals/史料/东方杂志/东方杂志1915年第12卷第2期.djvu` | 东方杂志1915年第12卷第2期.djvu |  |  | 否 |
| 137 | `Journals/史料/东方杂志/东方杂志1915年第12卷第3期.djvu` | 东方杂志1915年第12卷第3期.djvu |  |  | 否 |
| 138 | `Journals/史料/东方杂志/东方杂志1915年第12卷第4期.djvu` | 东方杂志1915年第12卷第4期.djvu |  |  | 否 |
| 139 | `Journals/史料/东方杂志/东方杂志1915年第12卷第5期.djvu` | 东方杂志1915年第12卷第5期.djvu |  |  | 否 |
| 140 | `Journals/史料/东方杂志/东方杂志1915年第12卷第6期.djvu` | 东方杂志1915年第12卷第6期.djvu |  |  | 否 |
| 141 | `Journals/史料/东方杂志/东方杂志1915年第12卷第7期.djvu` | 东方杂志1915年第12卷第7期.djvu |  |  | 否 |
| 142 | `Journals/史料/东方杂志/东方杂志1915年第12卷第8期.djvu` | 东方杂志1915年第12卷第8期.djvu |  |  | 否 |
| 143 | `Journals/史料/东方杂志/东方杂志1915年第12卷第9期.djvu` | 东方杂志1915年第12卷第9期.djvu |  |  | 否 |
| 144 | `Journals/史料/东方杂志/东方杂志1916年第13卷第11期.djvu` | 东方杂志1916年第13卷第11期.djvu |  |  | 否 |
| 145 | `Journals/史料/东方杂志/东方杂志1916年第13卷第12期.djvu` | 东方杂志1916年第13卷第12期.djvu |  |  | 否 |
| 146 | `Journals/史料/东方杂志/东方杂志1916年第13卷第1期.djvu` | 东方杂志1916年第13卷第1期.djvu |  |  | 否 |
| 147 | `Journals/史料/东方杂志/东方杂志1916年第13卷第2期.djvu` | 东方杂志1916年第13卷第2期.djvu |  |  | 否 |
| 148 | `Journals/史料/东方杂志/东方杂志1916年第13卷第3期.djvu` | 东方杂志1916年第13卷第3期.djvu |  |  | 否 |
| 149 | `Journals/史料/东方杂志/东方杂志1916年第13卷第4期.djvu` | 东方杂志1916年第13卷第4期.djvu |  |  | 否 |
| 150 | `Journals/史料/东方杂志/东方杂志1916年第13卷第5期.djvu` | 东方杂志1916年第13卷第5期.djvu |  |  | 否 |
| 151 | `Journals/史料/东方杂志/东方杂志1916年第13卷第6期.djvu` | 东方杂志1916年第13卷第6期.djvu |  |  | 否 |
| 152 | `Journals/史料/东方杂志/东方杂志1916年第13卷第7期.djvu` | 东方杂志1916年第13卷第7期.djvu |  |  | 否 |
| 153 | `Journals/史料/东方杂志/东方杂志1917年第14卷第10期.djvu` | 东方杂志1917年第14卷第10期.djvu |  |  | 否 |
| 154 | `Journals/史料/东方杂志/东方杂志1917年第14卷第11期.djvu` | 东方杂志1917年第14卷第11期.djvu |  |  | 否 |
| 155 | `Journals/史料/东方杂志/东方杂志1917年第14卷第12期.djvu` | 东方杂志1917年第14卷第12期.djvu |  |  | 否 |
| 156 | `Journals/史料/东方杂志/东方杂志1917年第14卷第1期.djvu` | 东方杂志1917年第14卷第1期.djvu |  |  | 否 |
| 157 | `Journals/史料/东方杂志/东方杂志1917年第14卷第2期.djvu` | 东方杂志1917年第14卷第2期.djvu |  |  | 否 |
| 158 | `Journals/史料/东方杂志/东方杂志1917年第14卷第3期.djvu` | 东方杂志1917年第14卷第3期.djvu |  |  | 否 |
| 159 | `Journals/史料/东方杂志/东方杂志1917年第14卷第4期.djvu` | 东方杂志1917年第14卷第4期.djvu |  |  | 否 |
| 160 | `Journals/史料/东方杂志/东方杂志1917年第14卷第5期.djvu` | 东方杂志1917年第14卷第5期.djvu |  |  | 否 |
| 161 | `Journals/史料/东方杂志/东方杂志1917年第14卷第6期.djvu` | 东方杂志1917年第14卷第6期.djvu |  |  | 否 |
| 162 | `Journals/史料/东方杂志/东方杂志1917年第14卷第7期.djvu` | 东方杂志1917年第14卷第7期.djvu |  |  | 否 |
| 163 | `Journals/史料/东方杂志/东方杂志1917年第14卷第9期.djvu` | 东方杂志1917年第14卷第9期.djvu |  |  | 否 |
| 164 | `Journals/史料/东方杂志/东方杂志1918年第15卷第10期.djvu` | 东方杂志1918年第15卷第10期.djvu |  |  | 否 |
| 165 | `Journals/史料/东方杂志/东方杂志1918年第15卷第11期.djvu` | 东方杂志1918年第15卷第11期.djvu |  |  | 否 |
| 166 | `Journals/史料/东方杂志/东方杂志1918年第15卷第12期.djvu` | 东方杂志1918年第15卷第12期.djvu |  |  | 否 |
| 167 | `Journals/史料/东方杂志/东方杂志1918年第15卷第1期.djvu` | 东方杂志1918年第15卷第1期.djvu |  |  | 否 |
| 168 | `Journals/史料/东方杂志/东方杂志1918年第15卷第4期.djvu` | 东方杂志1918年第15卷第4期.djvu |  |  | 否 |
| 169 | `Journals/史料/东方杂志/东方杂志1918年第15卷第5期.djvu` | 东方杂志1918年第15卷第5期.djvu |  |  | 否 |
| 170 | `Journals/史料/东方杂志/东方杂志1918年第15卷第6期.djvu` | 东方杂志1918年第15卷第6期.djvu |  |  | 否 |
| 171 | `Journals/史料/东方杂志/东方杂志1918年第15卷第7期.djvu` | 东方杂志1918年第15卷第7期.djvu |  |  | 否 |
| 172 | `Journals/史料/东方杂志/东方杂志1918年第15卷第8期.djvu` | 东方杂志1918年第15卷第8期.djvu |  |  | 否 |
| 173 | `Journals/史料/东方杂志/东方杂志1918年第15卷第9期.djvu` | 东方杂志1918年第15卷第9期.djvu |  |  | 否 |
| 174 | `Journals/史料/东方杂志/东方杂志1919年第16卷第10期.djvu` | 东方杂志1919年第16卷第10期.djvu |  |  | 否 |
| 175 | `Journals/史料/东方杂志/东方杂志1919年第16卷第11期.djvu` | 东方杂志1919年第16卷第11期.djvu |  |  | 否 |
| 176 | `Journals/史料/东方杂志/东方杂志1919年第16卷第12期.djvu` | 东方杂志1919年第16卷第12期.djvu |  |  | 否 |
| 177 | `Journals/史料/东方杂志/东方杂志1919年第16卷第1期.djvu` | 东方杂志1919年第16卷第1期.djvu |  |  | 否 |
| 178 | `Journals/史料/东方杂志/东方杂志1919年第16卷第2期.djvu` | 东方杂志1919年第16卷第2期.djvu |  |  | 否 |
| 179 | `Journals/史料/东方杂志/东方杂志1919年第16卷第3期.djvu` | 东方杂志1919年第16卷第3期.djvu |  |  | 否 |
| 180 | `Journals/史料/东方杂志/东方杂志1919年第16卷第4期.djvu` | 东方杂志1919年第16卷第4期.djvu |  |  | 否 |
| 181 | `Journals/史料/东方杂志/东方杂志1919年第16卷第5期.djvu` | 东方杂志1919年第16卷第5期.djvu |  |  | 否 |
| 182 | `Journals/史料/东方杂志/东方杂志1919年第16卷第6期.djvu` | 东方杂志1919年第16卷第6期.djvu |  |  | 否 |
| 183 | `Journals/史料/东方杂志/东方杂志1919年第16卷第8期.djvu` | 东方杂志1919年第16卷第8期.djvu |  |  | 否 |
| 184 | `Journals/史料/东方杂志/东方杂志1919年第16卷第9期.djvu` | 东方杂志1919年第16卷第9期.djvu |  |  | 否 |
| 185 | `Journals/史料/东方杂志/东方杂志1920年第17卷第10期.djvu` | 东方杂志1920年第17卷第10期.djvu |  |  | 否 |
| 186 | `Journals/史料/东方杂志/东方杂志1920年第17卷第11期.djvu` | 东方杂志1920年第17卷第11期.djvu |  |  | 否 |
| 187 | `Journals/史料/东方杂志/东方杂志1920年第17卷第12期.djvu` | 东方杂志1920年第17卷第12期.djvu |  |  | 否 |
| 188 | `Journals/史料/东方杂志/东方杂志1920年第17卷第13期.djvu` | 东方杂志1920年第17卷第13期.djvu |  |  | 否 |
| 189 | `Journals/史料/东方杂志/东方杂志1920年第17卷第14期.djvu` | 东方杂志1920年第17卷第14期.djvu |  |  | 否 |
| 190 | `Journals/史料/东方杂志/东方杂志1920年第17卷第15期.djvu` | 东方杂志1920年第17卷第15期.djvu |  |  | 否 |
| 191 | `Journals/史料/东方杂志/东方杂志1920年第17卷第16期.djvu` | 东方杂志1920年第17卷第16期.djvu |  |  | 否 |
| 192 | `Journals/史料/东方杂志/东方杂志1920年第17卷第17期.djvu` | 东方杂志1920年第17卷第17期.djvu |  |  | 否 |
| 193 | `Journals/史料/东方杂志/东方杂志1920年第17卷第18期.djvu` | 东方杂志1920年第17卷第18期.djvu |  |  | 否 |
| 194 | `Journals/史料/东方杂志/东方杂志1920年第17卷第19期.djvu` | 东方杂志1920年第17卷第19期.djvu |  |  | 否 |
| 195 | `Journals/史料/东方杂志/东方杂志1920年第17卷第1期.djvu` | 东方杂志1920年第17卷第1期.djvu |  |  | 否 |
| 196 | `Journals/史料/东方杂志/东方杂志1920年第17卷第20期.djvu` | 东方杂志1920年第17卷第20期.djvu |  |  | 否 |
| 197 | `Journals/史料/东方杂志/东方杂志1920年第17卷第21期.djvu` | 东方杂志1920年第17卷第21期.djvu |  |  | 否 |
| 198 | `Journals/史料/东方杂志/东方杂志1920年第17卷第22期.djvu` | 东方杂志1920年第17卷第22期.djvu |  |  | 否 |
| 199 | `Journals/史料/东方杂志/东方杂志1920年第17卷第23期.djvu` | 东方杂志1920年第17卷第23期.djvu |  |  | 否 |
| 200 | `Journals/史料/东方杂志/东方杂志1920年第17卷第24期.djvu` | 东方杂志1920年第17卷第24期.djvu |  |  | 否 |
| 201 | `Journals/史料/东方杂志/东方杂志1920年第17卷第2期.djvu` | 东方杂志1920年第17卷第2期.djvu |  |  | 否 |
| 202 | `Journals/史料/东方杂志/东方杂志1920年第17卷第3期.djvu` | 东方杂志1920年第17卷第3期.djvu |  |  | 否 |
| 203 | `Journals/史料/东方杂志/东方杂志1920年第17卷第4期.djvu` | 东方杂志1920年第17卷第4期.djvu |  |  | 否 |
| 204 | `Journals/史料/东方杂志/东方杂志1920年第17卷第5期.djvu` | 东方杂志1920年第17卷第5期.djvu |  |  | 否 |
| 205 | `Journals/史料/东方杂志/东方杂志1920年第17卷第6期.djvu` | 东方杂志1920年第17卷第6期.djvu |  |  | 否 |
| 206 | `Journals/史料/东方杂志/东方杂志1920年第17卷第7期.djvu` | 东方杂志1920年第17卷第7期.djvu |  |  | 否 |
| 207 | `Journals/史料/东方杂志/东方杂志1920年第17卷第8期.djvu` | 东方杂志1920年第17卷第8期.djvu |  |  | 否 |
| 208 | `Journals/史料/东方杂志/东方杂志1920年第17卷第9期.djvu` | 东方杂志1920年第17卷第9期.djvu |  |  | 否 |
| 209 | `Journals/史料/东方杂志/东方杂志1921年第18卷第10期.djvu` | 东方杂志1921年第18卷第10期.djvu |  |  | 否 |
| 210 | `Journals/史料/东方杂志/东方杂志1921年第18卷第11期.djvu` | 东方杂志1921年第18卷第11期.djvu |  |  | 否 |
| 211 | `Journals/史料/东方杂志/东方杂志1921年第18卷第12期.djvu` | 东方杂志1921年第18卷第12期.djvu |  |  | 否 |
| 212 | `Journals/史料/东方杂志/东方杂志1921年第18卷第13期.djvu` | 东方杂志1921年第18卷第13期.djvu |  |  | 否 |
| 213 | `Journals/史料/东方杂志/东方杂志1921年第18卷第14期.djvu` | 东方杂志1921年第18卷第14期.djvu |  |  | 否 |
| 214 | `Journals/史料/东方杂志/东方杂志1921年第18卷第15期.djvu` | 东方杂志1921年第18卷第15期.djvu |  |  | 否 |
| 215 | `Journals/史料/东方杂志/东方杂志1921年第18卷第16期.djvu` | 东方杂志1921年第18卷第16期.djvu |  |  | 否 |
| 216 | `Journals/史料/东方杂志/东方杂志1921年第18卷第17期.djvu` | 东方杂志1921年第18卷第17期.djvu |  |  | 否 |
| 217 | `Journals/史料/东方杂志/东方杂志1921年第18卷第1期.djvu` | 东方杂志1921年第18卷第1期.djvu |  |  | 否 |
| 218 | `Journals/史料/东方杂志/东方杂志1921年第18卷第20期.djvu` | 东方杂志1921年第18卷第20期.djvu |  |  | 否 |
| 219 | `Journals/史料/东方杂志/东方杂志1921年第18卷第21期.djvu` | 东方杂志1921年第18卷第21期.djvu |  |  | 否 |
| 220 | `Journals/史料/东方杂志/东方杂志1921年第18卷第22期.djvu` | 东方杂志1921年第18卷第22期.djvu |  |  | 否 |
| 221 | `Journals/史料/东方杂志/东方杂志1921年第18卷第23期.djvu` | 东方杂志1921年第18卷第23期.djvu |  |  | 否 |
| 222 | `Journals/史料/东方杂志/东方杂志1921年第18卷第2期.djvu` | 东方杂志1921年第18卷第2期.djvu |  |  | 否 |
| 223 | `Journals/史料/东方杂志/东方杂志1921年第18卷第3期.djvu` | 东方杂志1921年第18卷第3期.djvu |  |  | 否 |
| 224 | `Journals/史料/东方杂志/东方杂志1921年第18卷第4期.djvu` | 东方杂志1921年第18卷第4期.djvu |  |  | 否 |
| 225 | `Journals/史料/东方杂志/东方杂志1921年第18卷第5期.djvu` | 东方杂志1921年第18卷第5期.djvu |  |  | 否 |
| 226 | `Journals/史料/东方杂志/东方杂志1921年第18卷第6期.djvu` | 东方杂志1921年第18卷第6期.djvu |  |  | 否 |
| 227 | `Journals/史料/东方杂志/东方杂志1921年第18卷第7期.djvu` | 东方杂志1921年第18卷第7期.djvu |  |  | 否 |
| 228 | `Journals/史料/东方杂志/东方杂志1921年第18卷第8期.djvu` | 东方杂志1921年第18卷第8期.djvu |  |  | 否 |
| 229 | `Journals/史料/东方杂志/东方杂志1921年第18卷第9期.djvu` | 东方杂志1921年第18卷第9期.djvu |  |  | 否 |
| 230 | `Journals/史料/东方杂志/东方杂志1922年第19卷第10期.djvu` | 东方杂志1922年第19卷第10期.djvu |  |  | 否 |
| 231 | `Journals/史料/东方杂志/东方杂志1922年第19卷第11期.djvu` | 东方杂志1922年第19卷第11期.djvu |  |  | 否 |
| 232 | `Journals/史料/东方杂志/东方杂志1922年第19卷第12期.djvu` | 东方杂志1922年第19卷第12期.djvu |  |  | 否 |
| 233 | `Journals/史料/东方杂志/东方杂志1922年第19卷第13期.djvu` | 东方杂志1922年第19卷第13期.djvu |  |  | 否 |
| 234 | `Journals/史料/东方杂志/东方杂志1922年第19卷第14期.djvu` | 东方杂志1922年第19卷第14期.djvu |  |  | 否 |
| 235 | `Journals/史料/东方杂志/东方杂志1922年第19卷第15期.djvu` | 东方杂志1922年第19卷第15期.djvu |  |  | 否 |
| 236 | `Journals/史料/东方杂志/东方杂志1922年第19卷第16期.djvu` | 东方杂志1922年第19卷第16期.djvu |  |  | 否 |
| 237 | `Journals/史料/东方杂志/东方杂志1922年第19卷第17期.djvu` | 东方杂志1922年第19卷第17期.djvu |  |  | 否 |
| 238 | `Journals/史料/东方杂志/东方杂志1922年第19卷第18期.djvu` | 东方杂志1922年第19卷第18期.djvu |  |  | 否 |
| 239 | `Journals/史料/东方杂志/东方杂志1922年第19卷第19期.djvu` | 东方杂志1922年第19卷第19期.djvu |  |  | 否 |
| 240 | `Journals/史料/东方杂志/东方杂志1922年第19卷第1期.djvu` | 东方杂志1922年第19卷第1期.djvu |  |  | 否 |
| 241 | `Journals/史料/东方杂志/东方杂志1922年第19卷第20期.djvu` | 东方杂志1922年第19卷第20期.djvu |  |  | 否 |
| 242 | `Journals/史料/东方杂志/东方杂志1922年第19卷第23期.djvu` | 东方杂志1922年第19卷第23期.djvu |  |  | 否 |
| 243 | `Journals/史料/东方杂志/东方杂志1922年第19卷第24期.djvu` | 东方杂志1922年第19卷第24期.djvu |  |  | 否 |
| 244 | `Journals/史料/东方杂志/东方杂志1922年第19卷第2期.djvu` | 东方杂志1922年第19卷第2期.djvu |  |  | 否 |
| 245 | `Journals/史料/东方杂志/东方杂志1922年第19卷第3期.djvu` | 东方杂志1922年第19卷第3期.djvu |  |  | 否 |
| 246 | `Journals/史料/东方杂志/东方杂志1922年第19卷第4期.djvu` | 东方杂志1922年第19卷第4期.djvu |  |  | 否 |
| 247 | `Journals/史料/东方杂志/东方杂志1922年第19卷第5期.djvu` | 东方杂志1922年第19卷第5期.djvu |  |  | 否 |
| 248 | `Journals/史料/东方杂志/东方杂志1922年第19卷第6期.djvu` | 东方杂志1922年第19卷第6期.djvu |  |  | 否 |
| 249 | `Journals/史料/东方杂志/东方杂志1922年第19卷第7期.djvu` | 东方杂志1922年第19卷第7期.djvu |  |  | 否 |
| 250 | `Journals/史料/东方杂志/东方杂志1922年第19卷第8期.djvu` | 东方杂志1922年第19卷第8期.djvu |  |  | 否 |
| 251 | `Journals/史料/东方杂志/东方杂志1922年第19卷第9期.djvu` | 东方杂志1922年第19卷第9期.djvu |  |  | 否 |
| 252 | `Journals/史料/东方杂志/东方杂志1923年第20卷第10期.djvu` | 东方杂志1923年第20卷第10期.djvu |  |  | 否 |
| 253 | `Journals/史料/东方杂志/东方杂志1923年第20卷第11期.djvu` | 东方杂志1923年第20卷第11期.djvu |  |  | 否 |
| 254 | `Journals/史料/东方杂志/东方杂志1923年第20卷第12期.djvu` | 东方杂志1923年第20卷第12期.djvu |  |  | 否 |
| 255 | `Journals/史料/东方杂志/东方杂志1923年第20卷第13期.djvu` | 东方杂志1923年第20卷第13期.djvu |  |  | 否 |
| 256 | `Journals/史料/东方杂志/东方杂志1923年第20卷第14期.djvu` | 东方杂志1923年第20卷第14期.djvu |  |  | 否 |
| 257 | `Journals/史料/东方杂志/东方杂志1923年第20卷第17期.djvu` | 东方杂志1923年第20卷第17期.djvu |  |  | 否 |
| 258 | `Journals/史料/东方杂志/东方杂志1923年第20卷第18期.djvu` | 东方杂志1923年第20卷第18期.djvu |  |  | 否 |
| 259 | `Journals/史料/东方杂志/东方杂志1923年第20卷第19期.djvu` | 东方杂志1923年第20卷第19期.djvu |  |  | 否 |
| 260 | `Journals/史料/东方杂志/东方杂志1923年第20卷第1期.djvu` | 东方杂志1923年第20卷第1期.djvu |  |  | 否 |
| 261 | `Journals/史料/东方杂志/东方杂志1923年第20卷第20期.djvu` | 东方杂志1923年第20卷第20期.djvu |  |  | 否 |
| 262 | `Journals/史料/东方杂志/东方杂志1923年第20卷第21期.djvu` | 东方杂志1923年第20卷第21期.djvu |  |  | 否 |
| 263 | `Journals/史料/东方杂志/东方杂志1923年第20卷第22期.djvu` | 东方杂志1923年第20卷第22期.djvu |  |  | 否 |
| 264 | `Journals/史料/东方杂志/东方杂志1923年第20卷第23期.djvu` | 东方杂志1923年第20卷第23期.djvu |  |  | 否 |
| 265 | `Journals/史料/东方杂志/东方杂志1923年第20卷第24期.djvu` | 东方杂志1923年第20卷第24期.djvu |  |  | 否 |
| 266 | `Journals/史料/东方杂志/东方杂志1923年第20卷第2期.djvu` | 东方杂志1923年第20卷第2期.djvu |  |  | 否 |
| 267 | `Journals/史料/东方杂志/东方杂志1923年第20卷第3期.djvu` | 东方杂志1923年第20卷第3期.djvu |  |  | 否 |
| 268 | `Journals/史料/东方杂志/东方杂志1923年第20卷第4期.djvu` | 东方杂志1923年第20卷第4期.djvu |  |  | 否 |
| 269 | `Journals/史料/东方杂志/东方杂志1923年第20卷第5期.djvu` | 东方杂志1923年第20卷第5期.djvu |  |  | 否 |
| 270 | `Journals/史料/东方杂志/东方杂志1923年第20卷第6期.djvu` | 东方杂志1923年第20卷第6期.djvu |  |  | 否 |
| 271 | `Journals/史料/东方杂志/东方杂志1923年第20卷第7期.djvu` | 东方杂志1923年第20卷第7期.djvu |  |  | 否 |
| 272 | `Journals/史料/东方杂志/东方杂志1923年第20卷第8期.djvu` | 东方杂志1923年第20卷第8期.djvu |  |  | 否 |
| 273 | `Journals/史料/东方杂志/东方杂志1923年第20卷第9期.djvu` | 东方杂志1923年第20卷第9期.djvu |  |  | 否 |
| 274 | `Journals/史料/东方杂志/东方杂志1924年第21卷第13期.djvu` | 东方杂志1924年第21卷第13期.djvu |  |  | 否 |
| 275 | `Journals/史料/东方杂志/东方杂志1924年第21卷第14期.djvu` | 东方杂志1924年第21卷第14期.djvu |  |  | 否 |
| 276 | `Journals/史料/东方杂志/东方杂志1924年第21卷第15期.djvu` | 东方杂志1924年第21卷第15期.djvu |  |  | 否 |
| 277 | `Journals/史料/东方杂志/东方杂志1924年第21卷第16期.djvu` | 东方杂志1924年第21卷第16期.djvu |  |  | 否 |
| 278 | `Journals/史料/东方杂志/东方杂志1924年第21卷第17期.djvu` | 东方杂志1924年第21卷第17期.djvu |  |  | 否 |
| 279 | `Journals/史料/东方杂志/东方杂志1924年第21卷第18期.djvu` | 东方杂志1924年第21卷第18期.djvu |  |  | 否 |
| 280 | `Journals/史料/东方杂志/东方杂志1924年第21卷第19期.djvu` | 东方杂志1924年第21卷第19期.djvu |  |  | 否 |
| 281 | `Journals/史料/东方杂志/东方杂志1924年第21卷第20期.djvu` | 东方杂志1924年第21卷第20期.djvu |  |  | 否 |
| 282 | `Journals/史料/东方杂志/东方杂志1924年第21卷第21期.djvu` | 东方杂志1924年第21卷第21期.djvu |  |  | 否 |
| 283 | `Journals/史料/东方杂志/东方杂志1924年第21卷第22期.djvu` | 东方杂志1924年第21卷第22期.djvu |  |  | 否 |
| 284 | `Journals/史料/东方杂志/东方杂志1924年第21卷第23期.djvu` | 东方杂志1924年第21卷第23期.djvu |  |  | 否 |
| 285 | `Journals/史料/东方杂志/东方杂志1924年第21卷第24期.djvu` | 东方杂志1924年第21卷第24期.djvu |  |  | 否 |
| 286 | `Journals/史料/东方杂志/东方杂志1926年第23卷第13期.djvu` | 东方杂志1926年第23卷第13期.djvu |  |  | 否 |
| 287 | `Journals/史料/东方杂志/东方杂志1926年第23卷第14期.djvu` | 东方杂志1926年第23卷第14期.djvu |  |  | 否 |
| 288 | `Journals/史料/东方杂志/东方杂志1926年第23卷第15期.djvu` | 东方杂志1926年第23卷第15期.djvu |  |  | 否 |
| 289 | `Journals/史料/东方杂志/东方杂志1926年第23卷第16期.djvu` | 东方杂志1926年第23卷第16期.djvu |  |  | 否 |
| 290 | `Journals/史料/东方杂志/东方杂志1926年第23卷第17期.djvu` | 东方杂志1926年第23卷第17期.djvu |  |  | 否 |
| 291 | `Journals/史料/东方杂志/东方杂志1926年第23卷第18期.djvu` | 东方杂志1926年第23卷第18期.djvu |  |  | 否 |
| 292 | `Journals/史料/东方杂志/东方杂志1926年第23卷第19期.djvu` | 东方杂志1926年第23卷第19期.djvu |  |  | 否 |
| 293 | `Journals/史料/东方杂志/东方杂志1926年第23卷第1期.djvu` | 东方杂志1926年第23卷第1期.djvu |  |  | 否 |
| 294 | `Journals/史料/东方杂志/东方杂志1926年第23卷第20期.djvu` | 东方杂志1926年第23卷第20期.djvu |  |  | 否 |
| 295 | `Journals/史料/东方杂志/东方杂志1926年第23卷第21期.djvu` | 东方杂志1926年第23卷第21期.djvu |  |  | 否 |
| 296 | `Journals/史料/东方杂志/东方杂志1926年第23卷第22期.djvu` | 东方杂志1926年第23卷第22期.djvu |  |  | 否 |
| 297 | `Journals/史料/东方杂志/东方杂志1926年第23卷第23期.djvu` | 东方杂志1926年第23卷第23期.djvu |  |  | 否 |
| 298 | `Journals/史料/东方杂志/东方杂志1926年第23卷第24期.djvu` | 东方杂志1926年第23卷第24期.djvu |  |  | 否 |
| 299 | `Journals/史料/东方杂志/东方杂志1926年第23卷第2期.djvu` | 东方杂志1926年第23卷第2期.djvu |  |  | 否 |
| 300 | `Journals/史料/东方杂志/东方杂志1926年第23卷第3期.djvu` | 东方杂志1926年第23卷第3期.djvu |  |  | 否 |
| 301 | `Journals/史料/东方杂志/东方杂志1926年第23卷第4期.djvu` | 东方杂志1926年第23卷第4期.djvu |  |  | 否 |
| 302 | `Journals/史料/东方杂志/东方杂志1926年第23卷第5期.djvu` | 东方杂志1926年第23卷第5期.djvu |  |  | 否 |
| 303 | `Journals/史料/东方杂志/东方杂志1926年第23卷第6期.djvu` | 东方杂志1926年第23卷第6期.djvu |  |  | 否 |
| 304 | `Journals/史料/东方杂志/东方杂志1926年第23卷第7期.djvu` | 东方杂志1926年第23卷第7期.djvu |  |  | 否 |
| 305 | `Journals/史料/东方杂志/东方杂志1926年第23卷第8期.djvu` | 东方杂志1926年第23卷第8期.djvu |  |  | 否 |
| 306 | `Journals/史料/东方杂志/东方杂志1927年第24卷第10期.djvu` | 东方杂志1927年第24卷第10期.djvu |  |  | 否 |
| 307 | `Journals/史料/东方杂志/东方杂志1927年第24卷第11期.djvu` | 东方杂志1927年第24卷第11期.djvu |  |  | 否 |
| 308 | `Journals/史料/东方杂志/东方杂志1927年第24卷第12期.djvu` | 东方杂志1927年第24卷第12期.djvu |  |  | 否 |
| 309 | `Journals/史料/东方杂志/东方杂志1927年第24卷第13期.djvu` | 东方杂志1927年第24卷第13期.djvu |  |  | 否 |
| 310 | `Journals/史料/东方杂志/东方杂志1927年第24卷第14期.djvu` | 东方杂志1927年第24卷第14期.djvu |  |  | 否 |
| 311 | `Journals/史料/东方杂志/东方杂志1927年第24卷第15期.djvu` | 东方杂志1927年第24卷第15期.djvu |  |  | 否 |
| 312 | `Journals/史料/东方杂志/东方杂志1927年第24卷第16期.djvu` | 东方杂志1927年第24卷第16期.djvu |  |  | 否 |
| 313 | `Journals/史料/东方杂志/东方杂志1927年第24卷第17期.djvu` | 东方杂志1927年第24卷第17期.djvu |  |  | 否 |
| 314 | `Journals/史料/东方杂志/东方杂志1927年第24卷第18期.djvu` | 东方杂志1927年第24卷第18期.djvu |  |  | 否 |
| 315 | `Journals/史料/东方杂志/东方杂志1927年第24卷第19期.djvu` | 东方杂志1927年第24卷第19期.djvu |  |  | 否 |
| 316 | `Journals/史料/东方杂志/东方杂志1927年第24卷第1期.djvu` | 东方杂志1927年第24卷第1期.djvu |  |  | 否 |
| 317 | `Journals/史料/东方杂志/东方杂志1927年第24卷第20期.djvu` | 东方杂志1927年第24卷第20期.djvu |  |  | 否 |
| 318 | `Journals/史料/东方杂志/东方杂志1927年第24卷第21期.djvu` | 东方杂志1927年第24卷第21期.djvu |  |  | 否 |
| 319 | `Journals/史料/东方杂志/东方杂志1927年第24卷第22期.djvu` | 东方杂志1927年第24卷第22期.djvu |  |  | 否 |
| 320 | `Journals/史料/东方杂志/东方杂志1927年第24卷第23期.djvu` | 东方杂志1927年第24卷第23期.djvu |  |  | 否 |
| 321 | `Journals/史料/东方杂志/东方杂志1927年第24卷第24期.djvu` | 东方杂志1927年第24卷第24期.djvu |  |  | 否 |
| 322 | `Journals/史料/东方杂志/东方杂志1927年第24卷第2期.djvu` | 东方杂志1927年第24卷第2期.djvu |  |  | 否 |
| 323 | `Journals/史料/东方杂志/东方杂志1927年第24卷第3期.djvu` | 东方杂志1927年第24卷第3期.djvu |  |  | 否 |
| 324 | `Journals/史料/东方杂志/东方杂志1927年第24卷第4期.djvu` | 东方杂志1927年第24卷第4期.djvu |  |  | 否 |
| 325 | `Journals/史料/东方杂志/东方杂志1927年第24卷第5期.djvu` | 东方杂志1927年第24卷第5期.djvu |  |  | 否 |
| 326 | `Journals/史料/东方杂志/东方杂志1927年第24卷第6期.djvu` | 东方杂志1927年第24卷第6期.djvu |  |  | 否 |
| 327 | `Journals/史料/东方杂志/东方杂志1927年第24卷第7期.djvu` | 东方杂志1927年第24卷第7期.djvu |  |  | 否 |
| 328 | `Journals/史料/东方杂志/东方杂志1927年第24卷第8期.djvu` | 东方杂志1927年第24卷第8期.djvu |  |  | 否 |
| 329 | `Journals/史料/东方杂志/东方杂志1927年第24卷第9期.djvu` | 东方杂志1927年第24卷第9期.djvu |  |  | 否 |
| 330 | `Journals/史料/东方杂志/东方杂志1928年第25卷第10期.djvu` | 东方杂志1928年第25卷第10期.djvu |  |  | 否 |
| 331 | `Journals/史料/东方杂志/东方杂志1928年第25卷第11期.djvu` | 东方杂志1928年第25卷第11期.djvu |  |  | 否 |
| 332 | `Journals/史料/东方杂志/东方杂志1928年第25卷第12期.djvu` | 东方杂志1928年第25卷第12期.djvu |  |  | 否 |
| 333 | `Journals/史料/东方杂志/东方杂志1928年第25卷第13期.djvu` | 东方杂志1928年第25卷第13期.djvu |  |  | 否 |
| 334 | `Journals/史料/东方杂志/东方杂志1928年第25卷第14期.djvu` | 东方杂志1928年第25卷第14期.djvu |  |  | 否 |
| 335 | `Journals/史料/东方杂志/东方杂志1928年第25卷第15期.djvu` | 东方杂志1928年第25卷第15期.djvu |  |  | 否 |
| 336 | `Journals/史料/东方杂志/东方杂志1928年第25卷第16期.djvu` | 东方杂志1928年第25卷第16期.djvu |  |  | 否 |
| 337 | `Journals/史料/东方杂志/东方杂志1928年第25卷第17期.djvu` | 东方杂志1928年第25卷第17期.djvu |  |  | 否 |
| 338 | `Journals/史料/东方杂志/东方杂志1928年第25卷第18期.djvu` | 东方杂志1928年第25卷第18期.djvu |  |  | 否 |
| 339 | `Journals/史料/东方杂志/东方杂志1928年第25卷第19期.djvu` | 东方杂志1928年第25卷第19期.djvu |  |  | 否 |
| 340 | `Journals/史料/东方杂志/东方杂志1928年第25卷第1期.djvu` | 东方杂志1928年第25卷第1期.djvu |  |  | 否 |
| 341 | `Journals/史料/东方杂志/东方杂志1928年第25卷第20期.djvu` | 东方杂志1928年第25卷第20期.djvu |  |  | 否 |
| 342 | `Journals/史料/东方杂志/东方杂志1928年第25卷第21期.djvu` | 东方杂志1928年第25卷第21期.djvu |  |  | 否 |
| 343 | `Journals/史料/东方杂志/东方杂志1928年第25卷第22期.djvu` | 东方杂志1928年第25卷第22期.djvu |  |  | 否 |
| 344 | `Journals/史料/东方杂志/东方杂志1928年第25卷第24期.djvu` | 东方杂志1928年第25卷第24期.djvu |  |  | 否 |
| 345 | `Journals/史料/东方杂志/东方杂志1928年第25卷第2期.djvu` | 东方杂志1928年第25卷第2期.djvu |  |  | 否 |
| 346 | `Journals/史料/东方杂志/东方杂志1928年第25卷第3期.djvu` | 东方杂志1928年第25卷第3期.djvu |  |  | 否 |
| 347 | `Journals/史料/东方杂志/东方杂志1928年第25卷第4期.djvu` | 东方杂志1928年第25卷第4期.djvu |  |  | 否 |
| 348 | `Journals/史料/东方杂志/东方杂志1928年第25卷第5期.djvu` | 东方杂志1928年第25卷第5期.djvu |  |  | 否 |
| 349 | `Journals/史料/东方杂志/东方杂志1928年第25卷第6期.djvu` | 东方杂志1928年第25卷第6期.djvu |  |  | 否 |
| 350 | `Journals/史料/东方杂志/东方杂志1928年第25卷第7期.djvu` | 东方杂志1928年第25卷第7期.djvu |  |  | 否 |
| 351 | `Journals/史料/东方杂志/东方杂志1928年第25卷第8期.djvu` | 东方杂志1928年第25卷第8期.djvu |  |  | 否 |
| 352 | `Journals/史料/东方杂志/东方杂志1928年第25卷第9期.djvu` | 东方杂志1928年第25卷第9期.djvu |  |  | 否 |
| 353 | `Journals/史料/东方杂志/东方杂志1929年第26卷第10期.djvu` | 东方杂志1929年第26卷第10期.djvu |  |  | 否 |
| 354 | `Journals/史料/东方杂志/东方杂志1929年第26卷第11期.djvu` | 东方杂志1929年第26卷第11期.djvu |  |  | 否 |
| 355 | `Journals/史料/东方杂志/东方杂志1929年第26卷第12期.djvu` | 东方杂志1929年第26卷第12期.djvu |  |  | 否 |
| 356 | `Journals/史料/东方杂志/东方杂志1929年第26卷第13期.djvu` | 东方杂志1929年第26卷第13期.djvu |  |  | 否 |
| 357 | `Journals/史料/东方杂志/东方杂志1929年第26卷第14期.djvu` | 东方杂志1929年第26卷第14期.djvu |  |  | 否 |
| 358 | `Journals/史料/东方杂志/东方杂志1929年第26卷第15期.djvu` | 东方杂志1929年第26卷第15期.djvu |  |  | 否 |
| 359 | `Journals/史料/东方杂志/东方杂志1929年第26卷第16期.djvu` | 东方杂志1929年第26卷第16期.djvu |  |  | 否 |
| 360 | `Journals/史料/东方杂志/东方杂志1929年第26卷第17期.djvu` | 东方杂志1929年第26卷第17期.djvu |  |  | 否 |
| 361 | `Journals/史料/东方杂志/东方杂志1929年第26卷第18期.djvu` | 东方杂志1929年第26卷第18期.djvu |  |  | 否 |
| 362 | `Journals/史料/东方杂志/东方杂志1929年第26卷第19期.djvu` | 东方杂志1929年第26卷第19期.djvu |  |  | 否 |
| 363 | `Journals/史料/东方杂志/东方杂志1929年第26卷第20期.djvu` | 东方杂志1929年第26卷第20期.djvu |  |  | 否 |
| 364 | `Journals/史料/东方杂志/东方杂志1929年第26卷第21期.djvu` | 东方杂志1929年第26卷第21期.djvu |  |  | 否 |
| 365 | `Journals/史料/东方杂志/东方杂志1929年第26卷第22期.djvu` | 东方杂志1929年第26卷第22期.djvu |  |  | 否 |
| 366 | `Journals/史料/东方杂志/东方杂志1929年第26卷第23期.djvu` | 东方杂志1929年第26卷第23期.djvu |  |  | 否 |
| 367 | `Journals/史料/东方杂志/东方杂志1929年第26卷第24期.djvu` | 东方杂志1929年第26卷第24期.djvu |  |  | 否 |
| 368 | `Journals/史料/东方杂志/东方杂志1929年第26卷第5期.djvu` | 东方杂志1929年第26卷第5期.djvu |  |  | 否 |
| 369 | `Journals/史料/东方杂志/东方杂志1929年第26卷第6期.djvu` | 东方杂志1929年第26卷第6期.djvu |  |  | 否 |
| 370 | `Journals/史料/东方杂志/东方杂志1929年第26卷第7期.djvu` | 东方杂志1929年第26卷第7期.djvu |  |  | 否 |
| 371 | `Journals/史料/东方杂志/东方杂志1929年第26卷第8期.djvu` | 东方杂志1929年第26卷第8期.djvu |  |  | 否 |
| 372 | `Journals/史料/东方杂志/东方杂志1929年第26卷第9期.djvu` | 东方杂志1929年第26卷第9期.djvu |  |  | 否 |
| 373 | `Journals/史料/东方杂志/东方杂志1930年第27卷第10期.djvu` | 东方杂志1930年第27卷第10期.djvu |  |  | 否 |
| 374 | `Journals/史料/东方杂志/东方杂志1930年第27卷第11期.djvu` | 东方杂志1930年第27卷第11期.djvu |  |  | 否 |
| 375 | `Journals/史料/东方杂志/东方杂志1930年第27卷第12期.djvu` | 东方杂志1930年第27卷第12期.djvu |  |  | 否 |
| 376 | `Journals/史料/东方杂志/东方杂志1930年第27卷第13期.djvu` | 东方杂志1930年第27卷第13期.djvu |  |  | 否 |
| 377 | `Journals/史料/东方杂志/东方杂志1930年第27卷第14期.djvu` | 东方杂志1930年第27卷第14期.djvu |  |  | 否 |
| 378 | `Journals/史料/东方杂志/东方杂志1930年第27卷第15期.djvu` | 东方杂志1930年第27卷第15期.djvu |  |  | 否 |
| 379 | `Journals/史料/东方杂志/东方杂志1930年第27卷第16期.djvu` | 东方杂志1930年第27卷第16期.djvu |  |  | 否 |
| 380 | `Journals/史料/东方杂志/东方杂志1930年第27卷第17期.djvu` | 东方杂志1930年第27卷第17期.djvu |  |  | 否 |
| 381 | `Journals/史料/东方杂志/东方杂志1930年第27卷第18期.djvu` | 东方杂志1930年第27卷第18期.djvu |  |  | 否 |
| 382 | `Journals/史料/东方杂志/东方杂志1930年第27卷第19期.djvu` | 东方杂志1930年第27卷第19期.djvu |  |  | 否 |
| 383 | `Journals/史料/东方杂志/东方杂志1930年第27卷第1期.djvu` | 东方杂志1930年第27卷第1期.djvu |  |  | 否 |
| 384 | `Journals/史料/东方杂志/东方杂志1930年第27卷第20期.djvu` | 东方杂志1930年第27卷第20期.djvu |  |  | 否 |
| 385 | `Journals/史料/东方杂志/东方杂志1930年第27卷第21期.djvu` | 东方杂志1930年第27卷第21期.djvu |  |  | 否 |
| 386 | `Journals/史料/东方杂志/东方杂志1930年第27卷第22期.djvu` | 东方杂志1930年第27卷第22期.djvu |  |  | 否 |
| 387 | `Journals/史料/东方杂志/东方杂志1930年第27卷第24期.djvu` | 东方杂志1930年第27卷第24期.djvu |  |  | 否 |
| 388 | `Journals/史料/东方杂志/东方杂志1930年第27卷第2期.djvu` | 东方杂志1930年第27卷第2期.djvu |  |  | 否 |
| 389 | `Journals/史料/东方杂志/东方杂志1930年第27卷第3期.djvu` | 东方杂志1930年第27卷第3期.djvu |  |  | 否 |
| 390 | `Journals/史料/东方杂志/东方杂志1930年第27卷第4期.djvu` | 东方杂志1930年第27卷第4期.djvu |  |  | 否 |
| 391 | `Journals/史料/东方杂志/东方杂志1930年第27卷第5期.djvu` | 东方杂志1930年第27卷第5期.djvu |  |  | 否 |
| 392 | `Journals/史料/东方杂志/东方杂志1930年第27卷第6期.djvu` | 东方杂志1930年第27卷第6期.djvu |  |  | 否 |
| 393 | `Journals/史料/东方杂志/东方杂志1930年第27卷第7期.djvu` | 东方杂志1930年第27卷第7期.djvu |  |  | 否 |
| 394 | `Journals/史料/东方杂志/东方杂志1930年第27卷第8期.djvu` | 东方杂志1930年第27卷第8期.djvu |  |  | 否 |
| 395 | `Journals/史料/东方杂志/东方杂志1930年第27卷第9期.djvu` | 东方杂志1930年第27卷第9期.djvu |  |  | 否 |
| 396 | `Journals/史料/东方杂志/东方杂志1931年第28卷第10期.djvu` | 东方杂志1931年第28卷第10期.djvu |  |  | 否 |
| 397 | `Journals/史料/东方杂志/东方杂志1931年第28卷第11期.djvu` | 东方杂志1931年第28卷第11期.djvu |  |  | 否 |
| 398 | `Journals/史料/东方杂志/东方杂志1931年第28卷第12期.djvu` | 东方杂志1931年第28卷第12期.djvu |  |  | 否 |
| 399 | `Journals/史料/东方杂志/东方杂志1931年第28卷第13期.djvu` | 东方杂志1931年第28卷第13期.djvu |  |  | 否 |
| 400 | `Journals/史料/东方杂志/东方杂志1931年第28卷第14期.djvu` | 东方杂志1931年第28卷第14期.djvu |  |  | 否 |
| 401 | `Journals/史料/东方杂志/东方杂志1931年第28卷第15期.djvu` | 东方杂志1931年第28卷第15期.djvu |  |  | 否 |
| 402 | `Journals/史料/东方杂志/东方杂志1931年第28卷第16期.djvu` | 东方杂志1931年第28卷第16期.djvu |  |  | 否 |
| 403 | `Journals/史料/东方杂志/东方杂志1931年第28卷第17期.djvu` | 东方杂志1931年第28卷第17期.djvu |  |  | 否 |
| 404 | `Journals/史料/东方杂志/东方杂志1931年第28卷第18期.djvu` | 东方杂志1931年第28卷第18期.djvu |  |  | 否 |
| 405 | `Journals/史料/东方杂志/东方杂志1931年第28卷第19期.djvu` | 东方杂志1931年第28卷第19期.djvu |  |  | 否 |
| 406 | `Journals/史料/东方杂志/东方杂志1931年第28卷第1期.djvu` | 东方杂志1931年第28卷第1期.djvu |  |  | 否 |
| 407 | `Journals/史料/东方杂志/东方杂志1931年第28卷第20期.djvu` | 东方杂志1931年第28卷第20期.djvu |  |  | 否 |
| 408 | `Journals/史料/东方杂志/东方杂志1931年第28卷第21期.djvu` | 东方杂志1931年第28卷第21期.djvu |  |  | 否 |
| 409 | `Journals/史料/东方杂志/东方杂志1931年第28卷第22期.djvu` | 东方杂志1931年第28卷第22期.djvu |  |  | 否 |
| 410 | `Journals/史料/东方杂志/东方杂志1931年第28卷第23期.djvu` | 东方杂志1931年第28卷第23期.djvu |  |  | 否 |
| 411 | `Journals/史料/东方杂志/东方杂志1931年第28卷第24期.djvu` | 东方杂志1931年第28卷第24期.djvu |  |  | 否 |
| 412 | `Journals/史料/东方杂志/东方杂志1931年第28卷第2期.djvu` | 东方杂志1931年第28卷第2期.djvu |  |  | 否 |
| 413 | `Journals/史料/东方杂志/东方杂志1931年第28卷第3期.djvu` | 东方杂志1931年第28卷第3期.djvu |  |  | 否 |
| 414 | `Journals/史料/东方杂志/东方杂志1931年第28卷第4期.djvu` | 东方杂志1931年第28卷第4期.djvu |  |  | 否 |
| 415 | `Journals/史料/东方杂志/东方杂志1931年第28卷第5期.djvu` | 东方杂志1931年第28卷第5期.djvu |  |  | 否 |
| 416 | `Journals/史料/东方杂志/东方杂志1931年第28卷第6期.djvu` | 东方杂志1931年第28卷第6期.djvu |  |  | 否 |
| 417 | `Journals/史料/东方杂志/东方杂志1931年第28卷第7期.djvu` | 东方杂志1931年第28卷第7期.djvu |  |  | 否 |
| 418 | `Journals/史料/东方杂志/东方杂志1931年第28卷第8期.djvu` | 东方杂志1931年第28卷第8期.djvu |  |  | 否 |
| 419 | `Journals/史料/东方杂志/东方杂志1931年第28卷第9期.djvu` | 东方杂志1931年第28卷第9期.djvu |  |  | 否 |
| 420 | `Journals/史料/东方杂志/东方杂志1932年01月第29卷第1期.djvu` | 东方杂志1932年01月第29卷第1期.djvu |  |  | 否 |
| 421 | `Journals/史料/东方杂志/东方杂志1932年01月第29卷第2期.djvu` | 东方杂志1932年01月第29卷第2期.djvu |  |  | 否 |
| 422 | `Journals/史料/东方杂志/东方杂志1932年02月第29卷第3期.djvu` | 东方杂志1932年02月第29卷第3期.djvu |  |  | 否 |
| 423 | `Journals/史料/东方杂志/东方杂志1932年10月第29卷第4期复刊号.djvu` | 东方杂志1932年10月第29卷第4期复刊号.djvu |  |  | 否 |
| 424 | `Journals/史料/东方杂志/东方杂志1932年11月第29卷第5期.djvu` | 东方杂志1932年11月第29卷第5期.djvu |  |  | 否 |
| 425 | `Journals/史料/东方杂志/东方杂志1932年11月第29卷第6期.djvu` | 东方杂志1932年11月第29卷第6期.djvu |  |  | 否 |
| 426 | `Journals/史料/东方杂志/东方杂志1932年12月第29卷第7期.djvu` | 东方杂志1932年12月第29卷第7期.djvu |  |  | 否 |
| 427 | `Journals/史料/东方杂志/东方杂志1932年12月第29卷第8期.djvu` | 东方杂志1932年12月第29卷第8期.djvu |  |  | 否 |
| 428 | `Journals/史料/东方杂志/东方杂志1933年第30卷第10期.djvu` | 东方杂志1933年第30卷第10期.djvu |  |  | 否 |
| 429 | `Journals/史料/东方杂志/东方杂志1933年第30卷第11期.djvu` | 东方杂志1933年第30卷第11期.djvu |  |  | 否 |
| 430 | `Journals/史料/东方杂志/东方杂志1933年第30卷第12期.djvu` | 东方杂志1933年第30卷第12期.djvu |  |  | 否 |
| 431 | `Journals/史料/东方杂志/东方杂志1933年第30卷第14期.djvu` | 东方杂志1933年第30卷第14期.djvu |  |  | 否 |
| 432 | `Journals/史料/东方杂志/东方杂志1933年第30卷第15期.djvu` | 东方杂志1933年第30卷第15期.djvu |  |  | 否 |
| 433 | `Journals/史料/东方杂志/东方杂志1933年第30卷第16期.djvu` | 东方杂志1933年第30卷第16期.djvu |  |  | 否 |
| 434 | `Journals/史料/东方杂志/东方杂志1933年第30卷第17期.djvu` | 东方杂志1933年第30卷第17期.djvu |  |  | 否 |
| 435 | `Journals/史料/东方杂志/东方杂志1933年第30卷第18期.djvu` | 东方杂志1933年第30卷第18期.djvu |  |  | 否 |
| 436 | `Journals/史料/东方杂志/东方杂志1933年第30卷第19期.djvu` | 东方杂志1933年第30卷第19期.djvu |  |  | 否 |
| 437 | `Journals/史料/东方杂志/东方杂志1933年第30卷第20期.djvu` | 东方杂志1933年第30卷第20期.djvu |  |  | 否 |
| 438 | `Journals/史料/东方杂志/东方杂志1933年第30卷第21期.djvu` | 东方杂志1933年第30卷第21期.djvu |  |  | 否 |
| 439 | `Journals/史料/东方杂志/东方杂志1933年第30卷第22期.djvu` | 东方杂志1933年第30卷第22期.djvu |  |  | 否 |
| 440 | `Journals/史料/东方杂志/东方杂志1933年第30卷第23期.djvu` | 东方杂志1933年第30卷第23期.djvu |  |  | 否 |
| 441 | `Journals/史料/东方杂志/东方杂志1933年第30卷第24期.djvu` | 东方杂志1933年第30卷第24期.djvu |  |  | 否 |
| 442 | `Journals/史料/东方杂志/东方杂志1933年第30卷第7期.djvu` | 东方杂志1933年第30卷第7期.djvu |  |  | 否 |
| 443 | `Journals/史料/东方杂志/东方杂志1933年第30卷第8期.djvu` | 东方杂志1933年第30卷第8期.djvu |  |  | 否 |
| 444 | `Journals/史料/东方杂志/东方杂志1933年第30卷第9期.djvu` | 东方杂志1933年第30卷第9期.djvu |  |  | 否 |
| 445 | `Journals/史料/东方杂志/东方杂志1935年第32卷第10期.djvu` | 东方杂志1935年第32卷第10期.djvu |  |  | 否 |
| 446 | `Journals/史料/东方杂志/东方杂志1935年第32卷第11期.djvu` | 东方杂志1935年第32卷第11期.djvu |  |  | 否 |
| 447 | `Journals/史料/东方杂志/东方杂志1935年第32卷第12期.djvu` | 东方杂志1935年第32卷第12期.djvu |  |  | 否 |
| 448 | `Journals/史料/东方杂志/东方杂志1935年第32卷第2期.djvu` | 东方杂志1935年第32卷第2期.djvu |  |  | 否 |
| 449 | `Journals/史料/东方杂志/东方杂志1935年第32卷第3期.djvu` | 东方杂志1935年第32卷第3期.djvu |  |  | 否 |
| 450 | `Journals/史料/东方杂志/东方杂志1935年第32卷第4期.djvu` | 东方杂志1935年第32卷第4期.djvu |  |  | 否 |
| 451 | `Journals/史料/东方杂志/东方杂志1935年第32卷第5期.djvu` | 东方杂志1935年第32卷第5期.djvu |  |  | 否 |
| 452 | `Journals/史料/东方杂志/东方杂志1935年第32卷第6期.djvu` | 东方杂志1935年第32卷第6期.djvu |  |  | 否 |
| 453 | `Journals/史料/东方杂志/东方杂志1935年第32卷第7期.djvu` | 东方杂志1935年第32卷第7期.djvu |  |  | 否 |
| 454 | `Journals/史料/东方杂志/东方杂志1935年第32卷第8期.djvu` | 东方杂志1935年第32卷第8期.djvu |  |  | 否 |
| 455 | `Journals/史料/东方杂志/东方杂志1935年第32卷第9期.djvu` | 东方杂志1935年第32卷第9期.djvu |  |  | 否 |
| 456 | `Journals/史料/东方杂志/东方杂志1936年第33卷第13期.djvu` | 东方杂志1936年第33卷第13期.djvu |  |  | 否 |
| 457 | `Journals/史料/东方杂志/东方杂志1936年第33卷第14期.djvu` | 东方杂志1936年第33卷第14期.djvu |  |  | 否 |
| 458 | `Journals/史料/东方杂志/东方杂志1936年第33卷第15期.djvu` | 东方杂志1936年第33卷第15期.djvu |  |  | 否 |
| 459 | `Journals/史料/东方杂志/东方杂志1936年第33卷第16期.djvu` | 东方杂志1936年第33卷第16期.djvu |  |  | 否 |
| 460 | `Journals/史料/东方杂志/东方杂志1936年第33卷第17期.djvu` | 东方杂志1936年第33卷第17期.djvu |  |  | 否 |
| 461 | `Journals/史料/东方杂志/东方杂志1936年第33卷第18期.djvu` | 东方杂志1936年第33卷第18期.djvu |  |  | 否 |
| 462 | `Journals/史料/东方杂志/东方杂志1936年第33卷第19期.djvu` | 东方杂志1936年第33卷第19期.djvu |  |  | 否 |
| 463 | `Journals/史料/东方杂志/东方杂志1936年第33卷第20期.djvu` | 东方杂志1936年第33卷第20期.djvu |  |  | 否 |
| 464 | `Journals/史料/东方杂志/东方杂志1936年第33卷第21期.djvu` | 东方杂志1936年第33卷第21期.djvu |  |  | 否 |
| 465 | `Journals/史料/东方杂志/东方杂志1936年第33卷第22期.djvu` | 东方杂志1936年第33卷第22期.djvu |  |  | 否 |
| 466 | `Journals/史料/东方杂志/东方杂志1936年第33卷第23期.djvu` | 东方杂志1936年第33卷第23期.djvu |  |  | 否 |
| 467 | `Journals/史料/东方杂志/东方杂志1936年第33卷第24期.djvu` | 东方杂志1936年第33卷第24期.djvu |  |  | 否 |
| 468 | `Journals/史料/东方杂志/东方杂志1936年第33卷第2期.djvu` | 东方杂志1936年第33卷第2期.djvu |  |  | 否 |
| 469 | `Journals/史料/东方杂志/东方杂志1936年第33卷第3期.djvu` | 东方杂志1936年第33卷第3期.djvu |  |  | 否 |
| 470 | `Journals/史料/东方杂志/东方杂志1936年第33卷第4期.djvu` | 东方杂志1936年第33卷第4期.djvu |  |  | 否 |
| 471 | `Journals/史料/东方杂志/东方杂志1936年第33卷第5期.djvu` | 东方杂志1936年第33卷第5期.djvu |  |  | 否 |
| 472 | `Journals/史料/东方杂志/东方杂志1936年第33卷第6期.djvu` | 东方杂志1936年第33卷第6期.djvu |  |  | 否 |
| 473 | `Journals/史料/东方杂志/东方杂志1936年第33卷第8期.djvu` | 东方杂志1936年第33卷第8期.djvu |  |  | 否 |
| 474 | `Journals/史料/东方杂志/东方杂志1937年01月第34卷第2期.djvu` | 东方杂志1937年01月第34卷第2期.djvu |  |  | 否 |
| 475 | `Journals/史料/东方杂志/东方杂志1937年02月第34卷第3期.djvu` | 东方杂志1937年02月第34卷第3期.djvu |  |  | 否 |
| 476 | `Journals/史料/东方杂志/东方杂志1937年02月第34卷第4期.djvu` | 东方杂志1937年02月第34卷第4期.djvu |  |  | 否 |
| 477 | `Journals/史料/东方杂志/东方杂志1937年03月第34卷第5期.djvu` | 东方杂志1937年03月第34卷第5期.djvu |  |  | 否 |
| 478 | `Journals/史料/东方杂志/东方杂志1937年03月第34卷第6期.djvu` | 东方杂志1937年03月第34卷第6期.djvu |  |  | 否 |
| 479 | `Journals/史料/东方杂志/东方杂志1937年04月第34卷第7期.djvu` | 东方杂志1937年04月第34卷第7期.djvu |  |  | 否 |
| 480 | `Journals/史料/东方杂志/东方杂志1937年04月第34卷第8期.djvu` | 东方杂志1937年04月第34卷第8期.djvu |  |  | 否 |
| 481 | `Journals/史料/东方杂志/东方杂志1937年05月第34卷第10期.djvu` | 东方杂志1937年05月第34卷第10期.djvu |  |  | 否 |
| 482 | `Journals/史料/东方杂志/东方杂志1937年05月第34卷第9期.djvu` | 东方杂志1937年05月第34卷第9期.djvu |  |  | 否 |
| 483 | `Journals/史料/东方杂志/东方杂志1937年06月第34卷第11期.djvu` | 东方杂志1937年06月第34卷第11期.djvu |  |  | 否 |
| 484 | `Journals/史料/东方杂志/东方杂志1937年06月第34卷第12期.djvu` | 东方杂志1937年06月第34卷第12期.djvu |  |  | 否 |
| 485 | `Journals/史料/东方杂志/东方杂志1937年07月第34卷第14期.djvu` | 东方杂志1937年07月第34卷第14期.djvu |  |  | 否 |
| 486 | `Journals/史料/东方杂志/东方杂志1937年08月第34卷第15期.djvu` | 东方杂志1937年08月第34卷第15期.djvu |  |  | 否 |
| 487 | `Journals/史料/东方杂志/东方杂志1938年第35卷第10期.djvu` | 东方杂志1938年第35卷第10期.djvu |  |  | 否 |
| 488 | `Journals/史料/东方杂志/东方杂志1938年第35卷第12期.djvu` | 东方杂志1938年第35卷第12期.djvu |  |  | 否 |
| 489 | `Journals/史料/东方杂志/东方杂志1938年第35卷第14期.djvu` | 东方杂志1938年第35卷第14期.djvu |  |  | 否 |
| 490 | `Journals/史料/东方杂志/东方杂志1938年第35卷第15期.djvu` | 东方杂志1938年第35卷第15期.djvu |  |  | 否 |
| 491 | `Journals/史料/东方杂志/东方杂志1938年第35卷第16期.djvu` | 东方杂志1938年第35卷第16期.djvu |  |  | 否 |
| 492 | `Journals/史料/东方杂志/东方杂志1938年第35卷第18期.djvu` | 东方杂志1938年第35卷第18期.djvu |  |  | 否 |
| 493 | `Journals/史料/东方杂志/东方杂志1938年第35卷第19期.djvu` | 东方杂志1938年第35卷第19期.djvu |  |  | 否 |
| 494 | `Journals/史料/东方杂志/东方杂志1938年第35卷第1期.djvu` | 东方杂志1938年第35卷第1期.djvu |  |  | 否 |
| 495 | `Journals/史料/东方杂志/东方杂志1938年第35卷第20期.djvu` | 东方杂志1938年第35卷第20期.djvu |  |  | 否 |
| 496 | `Journals/史料/东方杂志/东方杂志1938年第35卷第21期.djvu` | 东方杂志1938年第35卷第21期.djvu |  |  | 否 |
| 497 | `Journals/史料/东方杂志/东方杂志1938年第35卷第23期.djvu` | 东方杂志1938年第35卷第23期.djvu |  |  | 否 |
| 498 | `Journals/史料/东方杂志/东方杂志1938年第35卷第24期.djvu` | 东方杂志1938年第35卷第24期.djvu |  |  | 否 |
| 499 | `Journals/史料/东方杂志/东方杂志1938年第35卷第2期.djvu` | 东方杂志1938年第35卷第2期.djvu |  |  | 否 |
| 500 | `Journals/史料/东方杂志/东方杂志1938年第35卷第3期.djvu` | 东方杂志1938年第35卷第3期.djvu |  |  | 否 |
| 501 | `Journals/史料/东方杂志/东方杂志1938年第35卷第4期.djvu` | 东方杂志1938年第35卷第4期.djvu |  |  | 否 |
| 502 | `Journals/史料/东方杂志/东方杂志1938年第35卷第5期.djvu` | 东方杂志1938年第35卷第5期.djvu |  |  | 否 |
| 503 | `Journals/史料/东方杂志/东方杂志1938年第35卷第6期.djvu` | 东方杂志1938年第35卷第6期.djvu |  |  | 否 |
| 504 | `Journals/史料/东方杂志/东方杂志1938年第35卷第7期.djvu` | 东方杂志1938年第35卷第7期.djvu |  |  | 否 |
| 505 | `Journals/史料/东方杂志/东方杂志1938年第35卷第8期.djvu` | 东方杂志1938年第35卷第8期.djvu |  |  | 否 |
| 506 | `Journals/史料/东方杂志/东方杂志1938年第35卷第9期.djvu` | 东方杂志1938年第35卷第9期.djvu |  |  | 否 |
| 507 | `Journals/史料/东方杂志/东方杂志1940年第37卷第10期.djvu` | 东方杂志1940年第37卷第10期.djvu |  |  | 否 |
| 508 | `Journals/史料/东方杂志/东方杂志1940年第37卷第12期.djvu` | 东方杂志1940年第37卷第12期.djvu |  |  | 否 |
| 509 | `Journals/史料/东方杂志/东方杂志1940年第37卷第13期.djvu` | 东方杂志1940年第37卷第13期.djvu |  |  | 否 |
| 510 | `Journals/史料/东方杂志/东方杂志1940年第37卷第14期.djvu` | 东方杂志1940年第37卷第14期.djvu |  |  | 否 |
| 511 | `Journals/史料/东方杂志/东方杂志1940年第37卷第15期.djvu` | 东方杂志1940年第37卷第15期.djvu |  |  | 否 |
| 512 | `Journals/史料/东方杂志/东方杂志1940年第37卷第16期.djvu` | 东方杂志1940年第37卷第16期.djvu |  |  | 否 |
| 513 | `Journals/史料/东方杂志/东方杂志1940年第37卷第17期.djvu` | 东方杂志1940年第37卷第17期.djvu |  |  | 否 |
| 514 | `Journals/史料/东方杂志/东方杂志1940年第37卷第18期.djvu` | 东方杂志1940年第37卷第18期.djvu |  |  | 否 |
| 515 | `Journals/史料/东方杂志/东方杂志1940年第37卷第19期.djvu` | 东方杂志1940年第37卷第19期.djvu |  |  | 否 |
| 516 | `Journals/史料/东方杂志/东方杂志1940年第37卷第20期.djvu` | 东方杂志1940年第37卷第20期.djvu |  |  | 否 |
| 517 | `Journals/史料/东方杂志/东方杂志1940年第37卷第21期.djvu` | 东方杂志1940年第37卷第21期.djvu |  |  | 否 |
| 518 | `Journals/史料/东方杂志/东方杂志1940年第37卷第22期.djvu` | 东方杂志1940年第37卷第22期.djvu |  |  | 否 |
| 519 | `Journals/史料/东方杂志/东方杂志1940年第37卷第23期.djvu` | 东方杂志1940年第37卷第23期.djvu |  |  | 否 |
| 520 | `Journals/史料/东方杂志/东方杂志1940年第37卷第24期.djvu` | 东方杂志1940年第37卷第24期.djvu |  |  | 否 |
| 521 | `Journals/史料/东方杂志/东方杂志1940年第37卷第2期.djvu` | 东方杂志1940年第37卷第2期.djvu |  |  | 否 |
| 522 | `Journals/史料/东方杂志/东方杂志1940年第37卷第3期.djvu` | 东方杂志1940年第37卷第3期.djvu |  |  | 否 |
| 523 | `Journals/史料/东方杂志/东方杂志1940年第37卷第4期.djvu` | 东方杂志1940年第37卷第4期.djvu |  |  | 否 |
| 524 | `Journals/史料/东方杂志/东方杂志1940年第37卷第5期.djvu` | 东方杂志1940年第37卷第5期.djvu |  |  | 否 |
| 525 | `Journals/史料/东方杂志/东方杂志1940年第37卷第6期.djvu` | 东方杂志1940年第37卷第6期.djvu |  |  | 否 |
| 526 | `Journals/史料/东方杂志/东方杂志1940年第37卷第7期.djvu` | 东方杂志1940年第37卷第7期.djvu |  |  | 否 |
| 527 | `Journals/史料/东方杂志/东方杂志1940年第37卷第8期.djvu` | 东方杂志1940年第37卷第8期.djvu |  |  | 否 |
| 528 | `Journals/史料/东方杂志/东方杂志1941年第38卷第12期.djvu` | 东方杂志1941年第38卷第12期.djvu |  |  | 否 |
| 529 | `Journals/史料/东方杂志/东方杂志1941年第38卷第13期.djvu` | 东方杂志1941年第38卷第13期.djvu |  |  | 否 |
| 530 | `Journals/史料/东方杂志/东方杂志1941年第38卷第14期.djvu` | 东方杂志1941年第38卷第14期.djvu |  |  | 否 |
| 531 | `Journals/史料/东方杂志/东方杂志1941年第38卷第15期.djvu` | 东方杂志1941年第38卷第15期.djvu |  |  | 否 |
| 532 | `Journals/史料/东方杂志/东方杂志1941年第38卷第16期.djvu` | 东方杂志1941年第38卷第16期.djvu |  |  | 否 |
| 533 | `Journals/史料/东方杂志/东方杂志1941年第38卷第17期.djvu` | 东方杂志1941年第38卷第17期.djvu |  |  | 否 |
| 534 | `Journals/史料/东方杂志/东方杂志1941年第38卷第18期.djvu` | 东方杂志1941年第38卷第18期.djvu |  |  | 否 |
| 535 | `Journals/史料/东方杂志/东方杂志1941年第38卷第19期.djvu` | 东方杂志1941年第38卷第19期.djvu |  |  | 否 |
| 536 | `Journals/史料/东方杂志/东方杂志1941年第38卷第1期.djvu` | 东方杂志1941年第38卷第1期.djvu |  |  | 否 |
| 537 | `Journals/史料/东方杂志/东方杂志1941年第38卷第20期.djvu` | 东方杂志1941年第38卷第20期.djvu |  |  | 否 |
| 538 | `Journals/史料/东方杂志/东方杂志1941年第38卷第21期.djvu` | 东方杂志1941年第38卷第21期.djvu |  |  | 否 |
| 539 | `Journals/史料/东方杂志/东方杂志1941年第38卷第3期.djvu` | 东方杂志1941年第38卷第3期.djvu |  |  | 否 |
| 540 | `Journals/史料/东方杂志/东方杂志1941年第38卷第4期.djvu` | 东方杂志1941年第38卷第4期.djvu |  |  | 否 |
| 541 | `Journals/史料/东方杂志/东方杂志1941年第38卷第7期.djvu` | 东方杂志1941年第38卷第7期.djvu |  |  | 否 |
| 542 | `Journals/史料/东方杂志/东方杂志1941年第38卷第8期.djvu` | 东方杂志1941年第38卷第8期.djvu |  |  | 否 |
| 543 | `Journals/史料/东方杂志/东方杂志1941年第38卷第9期.djvu` | 东方杂志1941年第38卷第9期.djvu |  |  | 否 |
| 544 | `Journals/史料/东方杂志/东方杂志1944年第40卷第10期.djvu` | 东方杂志1944年第40卷第10期.djvu |  |  | 否 |
| 545 | `Journals/史料/东方杂志/东方杂志1944年第40卷第11期.djvu` | 东方杂志1944年第40卷第11期.djvu |  |  | 否 |
| 546 | `Journals/史料/东方杂志/东方杂志1944年第40卷第12期.djvu` | 东方杂志1944年第40卷第12期.djvu |  |  | 否 |
| 547 | `Journals/史料/东方杂志/东方杂志1944年第40卷第21期.djvu` | 东方杂志1944年第40卷第21期.djvu |  |  | 否 |
| 548 | `Journals/史料/东方杂志/东方杂志1944年第40卷第23期.djvu` | 东方杂志1944年第40卷第23期.djvu |  |  | 否 |
| 549 | `Journals/史料/东方杂志/东方杂志1944年第40卷第24期.djvu` | 东方杂志1944年第40卷第24期.djvu |  |  | 否 |
| 550 | `Journals/史料/东方杂志/东方杂志1944年第40卷第3期.djvu` | 东方杂志1944年第40卷第3期.djvu |  |  | 否 |
| 551 | `Journals/史料/东方杂志/东方杂志1944年第40卷第6期.djvu` | 东方杂志1944年第40卷第6期.djvu |  |  | 否 |
| 552 | `Journals/史料/东方杂志/东方杂志1944年第40卷第7期.djvu` | 东方杂志1944年第40卷第7期.djvu |  |  | 否 |
| 553 | `Journals/史料/东方杂志/东方杂志1944年第40卷第8期.djvu` | 东方杂志1944年第40卷第8期.djvu |  |  | 否 |
| 554 | `Journals/史料/东方杂志/东方杂志1944年第40卷第9期.djvu` | 东方杂志1944年第40卷第9期.djvu |  |  | 否 |
| 555 | `Journals/史料/东方杂志/东方杂志1946年第42卷第10期.djvu` | 东方杂志1946年第42卷第10期.djvu |  |  | 否 |
| 556 | `Journals/史料/东方杂志/东方杂志1946年第42卷第11期.djvu` | 东方杂志1946年第42卷第11期.djvu |  |  | 否 |
| 557 | `Journals/史料/东方杂志/东方杂志1946年第42卷第12期.djvu` | 东方杂志1946年第42卷第12期.djvu |  |  | 否 |
| 558 | `Journals/史料/东方杂志/东方杂志1946年第42卷第14期.djvu` | 东方杂志1946年第42卷第14期.djvu |  |  | 否 |
| 559 | `Journals/史料/东方杂志/东方杂志1946年第42卷第15期.djvu` | 东方杂志1946年第42卷第15期.djvu |  |  | 否 |
| 560 | `Journals/史料/东方杂志/东方杂志1946年第42卷第16期.djvu` | 东方杂志1946年第42卷第16期.djvu |  |  | 否 |
| 561 | `Journals/史料/东方杂志/东方杂志1946年第42卷第17期.djvu` | 东方杂志1946年第42卷第17期.djvu |  |  | 否 |
| 562 | `Journals/史料/东方杂志/东方杂志1946年第42卷第1期.djvu` | 东方杂志1946年第42卷第1期.djvu |  |  | 否 |
| 563 | `Journals/史料/东方杂志/东方杂志1946年第42卷第2期.djvu` | 东方杂志1946年第42卷第2期.djvu |  |  | 否 |
| 564 | `Journals/史料/东方杂志/东方杂志1946年第42卷第3期.djvu` | 东方杂志1946年第42卷第3期.djvu |  |  | 否 |
| 565 | `Journals/史料/东方杂志/东方杂志1946年第42卷第4期.djvu` | 东方杂志1946年第42卷第4期.djvu |  |  | 否 |
| 566 | `Journals/史料/东方杂志/东方杂志1946年第42卷第5期.djvu` | 东方杂志1946年第42卷第5期.djvu |  |  | 否 |
| 567 | `Journals/史料/东方杂志/东方杂志1946年第42卷第6期.djvu` | 东方杂志1946年第42卷第6期.djvu |  |  | 否 |
| 568 | `Journals/史料/东方杂志/东方杂志1946年第42卷第7期.djvu` | 东方杂志1946年第42卷第7期.djvu |  |  | 否 |
| 569 | `Journals/史料/东方杂志/东方杂志1946年第42卷第8期.djvu` | 东方杂志1946年第42卷第8期.djvu |  |  | 否 |
| 570 | `Journals/史料/东方杂志/东方杂志1946年第42卷第9期.djvu` | 东方杂志1946年第42卷第9期.djvu |  |  | 否 |
| 571 | `Journals/史料/东方杂志/东方杂志1947年第43卷第3期.djvu` | 东方杂志1947年第43卷第3期.djvu |  |  | 否 |
| 572 | `Journals/史料/东方杂志/东方杂志1947年第43卷第4期.djvu` | 东方杂志1947年第43卷第4期.djvu |  |  | 否 |
| 573 | `Journals/史料/东方杂志/东方杂志1948年第44卷第10期.djvu` | 东方杂志1948年第44卷第10期.djvu |  |  | 否 |
| 574 | `Journals/史料/东方杂志/东方杂志1948年第44卷第11期.djvu` | 东方杂志1948年第44卷第11期.djvu |  |  | 否 |
| 575 | `Journals/史料/东方杂志/东方杂志1948年第44卷第12期.djvu` | 东方杂志1948年第44卷第12期.djvu |  |  | 否 |
| 576 | `Journals/史料/东方杂志/东方杂志1948年第44卷第2期.djvu` | 东方杂志1948年第44卷第2期.djvu |  |  | 否 |
| 577 | `Journals/史料/东方杂志/东方杂志1948年第44卷第8期.djvu` | 东方杂志1948年第44卷第8期.djvu |  |  | 否 |

## Authors/（391 行）

| # | 原路径 | 文件名 | 标题 | 作者 | 父目录在盘 |
|---:|---|---|---|---|---|
| 1 | `Authors/Adorno, Theodor/Adorno, Theodor W._Lonitz, Henri - Towards a theory of musical reproduction_ notes, a draft, and` | Adorno, Theodor W._Lonitz, Henri - Towards a theory of musical reproduction_ notes, a draft, and two schemata-Polity (20 | Towards a theory of musical reproduction_ notes, a draft, and two schemata | Theodor W. Adorno\|Henri Lonitz | 否 |
| 2 | `Authors/Adorno, Theodor/Dialektik der Aufklarung_ philo - Theodor W. Adorno.pdf` | Dialektik der Aufklarung_ philo - Theodor W. Adorno.pdf | Dialektik der Aufklarung_ philo | Theodor W. Adorno | 否 |
| 3 | `Authors/Adorno, Theodor/Fou Ding De Bian Zheng Fa - Theodor W. Adorno.pdf` | Fou Ding De Bian Zheng Fa - Theodor W. Adorno.pdf | Fou Ding De Bian Zheng Fa | Theodor W. Adorno | 否 |
| 4 | `Authors/Adorno, Theodor/MINIMA MORALIA REFLECTIONS ON A - Theodor W. Adorno.pdf` | MINIMA MORALIA REFLECTIONS ON A - Theodor W. Adorno.pdf | MINIMA MORALIA REFLECTIONS ON A | Theodor W. Adorno | 否 |
| 5 | `Authors/Agamben, Giorgio/Giorgio Agamben--What-Is-an-Apparatus-and-Other-Essays-.pdf` | Giorgio Agamben--What-Is-an-Apparatus-and-Other-Essays-.pdf | What-Is-an-Apparatus-and-Other-Essays | Giorgio Agamben | 否 |
| 6 | `Authors/Alliez, Eric/Signature Of The World Or, What Is Deleuze And Guattari's Philosophy -Continuum (2004), The.pdf` | Signature Of The World Or, What Is Deleuze And Guattari's Philosophy -Continuum (2004), The.pdf | Signature Of The World Or, What Is Deleuze And Guattari's Philosophy | Eric Alliez | 否 |
| 7 | `Authors/Arendt, Hannah/Arendt Hannah The Origins of Totalitarianism 1979.pdf` | Arendt Hannah The Origins of Totalitarianism 1979.pdf | The Origins of Totalitarianism 1979 | Hannah Arendt | 否 |
| 8 | `Authors/Ascott, Roy/Roy Ascott - Telematic Embrace.pdf` | Roy Ascott - Telematic Embrace.pdf | Telematic Embrace | Roy Ascott | 否 |
| 9 | `Authors/Ascott, Roy/未来就是现在 艺术，技术和意识 （英）罗伊·阿斯科特著；袁小潆编；周凌，任爱凡译.pdf` | 未来就是现在 艺术，技术和意识 （英）罗伊·阿斯科特著；袁小潆编；周凌，任爱凡译.pdf | 未来就是现在 艺术，技术和意识 | Roy Ascott | 否 |
| 10 | `Authors/Bachelard, Gaston/巴什拉：空间的诗学（中文版）.pdf` | 巴什拉：空间的诗学（中文版）.pdf | 巴什拉：空间的诗学（中文版） | Gaston Bachelard | 否 |
| 11 | `Authors/Badiou, Alain/Alain Badiou - Being and Event-Continuum (2007).pdf` | Alain Badiou - Being and Event-Continuum (2007).pdf | Being and Event | Alain Badiou | 否 |
| 12 | `Authors/Badiou, Alain/Alain Badiou - The True Life.pdf` | Alain Badiou - The True Life.pdf | The True Life | Alain Badiou | 否 |
| 13 | `Authors/Badiou, Alain/Badiou Alain - Theory of the Subject.pdf` | Badiou Alain - Theory of the Subject.pdf | Theory of the Subject | Alain Badiou | 否 |
| 14 | `Authors/Badiou, Alain/In Praise of Love_Alain Badiou, Nicolas Truong, Translated by Peter Bush.pdf` | In Praise of Love_Alain Badiou, Nicolas Truong, Translated by Peter Bush.pdf | In Praise of Love | Alain Badiou | 否 |
| 15 | `Authors/Badiou, Alain/[Alain_Badiou]_Handbook_of_Inaesthetics_(Meridian(BookZZ.org)(cut).pdf` | [Alain_Badiou]_Handbook_of_Inaesthetics_(Meridian(BookZZ.org)(cut).pdf | Handbook_of_Inaesthetics | Alain Badiou | 否 |
| 16 | `Authors/Badiou, Alain/存在与事件_阿兰·巴迪欧.pdf` | 存在与事件_阿兰·巴迪欧.pdf | 存在与事件 | Alain Badiou | 否 |
| 17 | `Authors/Barthes, Roland/Myth Today_Barthes Roland..pdf` | Myth Today_Barthes Roland..pdf | Myth Today | Roland Barthes | 否 |
| 18 | `Authors/Barthes, Roland/Mythologies_Roland Barthes, Annette Lavers.pdf` | Mythologies_Roland Barthes, Annette Lavers.pdf | Mythologies | Roland Barthes | 否 |
| 19 | `Authors/Barthes, Roland/明室 摄影纵横谈 (无) (Z-Library).pdf` | 明室 摄影纵横谈 (无) (Z-Library).pdf | 明室 摄影纵横谈 | Roland Barthes | 否 |
| 20 | `Authors/Barthes, Roland/神话修辞术 - Roland Barthes.pdf` | 神话修辞术 - Roland Barthes.pdf | 神话修辞术 | Roland Barthes | 否 |
| 21 | `Authors/Bataille, Georges/Accursed Share Volume 1 Consumption, The.pdf` | Accursed Share Volume 1 Consumption, The.pdf | Accursed Share Volume 1 Consumption, The | Georges Bataille | 否 |
| 22 | `Authors/Bataille, Georges/Accursed Share, Vols. 2 and 3 The History of Eroticism and Sovereignty, The.pdf` | Accursed Share, Vols. 2 and 3 The History of Eroticism and Sovereignty, The.pdf | Accursed Share, Vols. 2 and 3 The History of Eroticism and Sovereignty, The | Georges Bataille | 否 |
| 23 | `Authors/Bataille, Georges/Bataille-La part maudite -precede de La Notion de Depense.pdf` | Bataille-La part maudite -precede de La Notion de Depense.pdf | La part maudite -precede de La Notion de Depense | Georges Bataille | 否 |
| 24 | `Authors/Bataille, Georges/Erotism Death and Sensuality.pdf` | Erotism Death and Sensuality.pdf | Erotism Death and Sensuality | Georges Bataille | 否 |
| 25 | `Authors/Bataille, Georges/Georges Bataille-Theory of Religion -Zone (1992).pdf` | Georges Bataille-Theory of Religion -Zone (1992).pdf | Theory of Religion | Georges Bataille | 否 |
| 26 | `Authors/Bataille, Georges/Impossible, The.pdf` | Impossible, The.pdf | Impossible, The | Georges Bataille | 否 |
| 27 | `Authors/Bataille, Georges/Unfinished System of Nonknowledge.pdf` | Unfinished System of Nonknowledge.pdf | Unfinished System of Nonknowledge | Georges Bataille | 否 |
| 28 | `Authors/Bateson, Gregory/Gregory Bateson - Mind and Nature.pdf` | Gregory Bateson - Mind and Nature.pdf | Mind and Nature | Gregory Bateson | 否 |
| 29 | `Authors/Bateson, Gregory/Gregory Bateson - Steps to an Ecology of Mind-Ballantine Books (1977).pdf` | Gregory Bateson - Steps to an Ecology of Mind-Ballantine Books (1977).pdf | Steps to an Ecology of Mind | Gregory Bateson | 否 |
| 30 | `Authors/Bateson, Gregory/心灵与自然 贝特森.pdf` | 心灵与自然 贝特森.pdf | 心灵与自然 贝特森 | Gregory Bateson | 否 |
| 31 | `Authors/Bauman, Richard/Explorations_in_the_Ethnography_of_Speaking_Joel_Sherzer;_Richard_Bauman.pdf` | Explorations_in_the_Ethnography_of_Speaking_Joel_Sherzer;_Richard_Bauman.pdf | Explorations_in_the_Ethnography_of_Speaking_Joel_Sherzer | Richard Bauman | 否 |
| 32 | `Authors/Beller, Jonathan/Jonathan Beller - The Message Is Murder_ Substrates of Computational Capital-Pluto Press (2018)` | Jonathan Beller - The Message Is Murder_ Substrates of Computational Capital-Pluto Press (2018).pdf | The Message Is Murder_ Substrates of Computational Capital | Jonathan Beller | 否 |
| 33 | `Authors/Benjamin, Walter/Band 2 AufsACURtze, Essays, VortrACURge. 3 TeilbACURnde -Suhrkamp (1977).pdf` | Band 2 AufsACURtze, Essays, VortrACURge. 3 TeilbACURnde -Suhrkamp (1977).pdf | Band 2 AufsACURtze, Essays, VortrACURge. 3 TeilbACURnde -Suhrkamp | Walter Benjamin | 否 |
| 34 | `Authors/Benjamin, Walter/本雅明文选-中国社会科学出版社 (2011).pdf` | 本雅明文选-中国社会科学出版社 (2011).pdf | 本雅明文选-中国社会科学出版社 | Walter Benjamin | 否 |
| 35 | `Authors/Berry, David M/Copy, Rip, Burn The Politics of Copyleft and Open Source-Pluto Press (2008) - Berry David.pdf` | Copy, Rip, Burn The Politics of Copyleft and Open Source-Pluto Press (2008) - Berry David.pdf | Copy, Rip, Burn The Politics of Copyleft and Open Source | Berry David | 否 |
| 36 | `Authors/Blanchot, Maurice/无尽的谈话_[法]_莫里斯·布朗肖_著;_尉光吉_译.pdf` | 无尽的谈话_[法]_莫里斯·布朗肖_著;_尉光吉_译.pdf | 无尽的谈话 | Maurice Blanchot | 否 |
| 37 | `Authors/Boden, Margaret/Creativity and Art Three Roads - A. Boden Margaret.pdf` | Creativity and Art Three Roads - A. Boden Margaret.pdf | Creativity and Art Three Roads | A. Boden Margaret | 否 |
| 38 | `Authors/Boden, Margaret/Margaret A. Boden - Fingers to Digits From.pdf` | Margaret A. Boden - Fingers to Digits From.pdf | A. Boden | Margaret Boden | 否 |
| 39 | `Authors/Bonitz, Hermann/Hermann Bonitz, Eduard Wellmann - Metaphysik des Aristoteles (deutsche Übersetzung)-GEORG REIMER` | Hermann Bonitz, Eduard Wellmann - Metaphysik des Aristoteles (deutsche Übersetzung)-GEORG REIMER (1890).pdf | Metaphysik des Aristoteles | Hermann Bonitz\|Eduard Wellmann | 否 |
| 40 | `Authors/Bourdieu, Pierre/Logic Of Practice_Bourdieu.pdf` | Logic Of Practice_Bourdieu.pdf | Logic Of Practice | Pierre Bourdieu | 否 |
| 41 | `Authors/Bourdieu, Pierre/The Logic of Practice_Pierre Bourdieu et al., Richard Nice.pdf` | The Logic of Practice_Pierre Bourdieu et al., Richard Nice.pdf | The Logic of Practice | Pierre Bourdieu | 否 |
| 42 | `Authors/Brecht, Bertolt/Brecht SCHRIFTEN ZUR LITERATUR UND KUNST Band I.pdf` | Brecht SCHRIFTEN ZUR LITERATUR UND KUNST Band I.pdf | SCHRIFTEN ZUR LITERATUR UND KUNST Band I | Bertolt Brecht | 否 |
| 43 | `Authors/Brecht, Bertolt/Brecht SCHRIFTEN ZUR LITERATUR UND KUNST Band II.pdf` | Brecht SCHRIFTEN ZUR LITERATUR UND KUNST Band II.pdf | SCHRIFTEN ZUR LITERATUR UND KUNST Band II | Bertolt Brecht | 否 |
| 44 | `Authors/Butterfield, Herbert/Herbert Butterfield - The Origins of Modern Science, 1300–1800-The Free Press (1965).pdf` | Herbert Butterfield - The Origins of Modern Science, 1300–1800-The Free Press (1965).pdf | The Origins of Modern Science, 1300–1800 | Herbert Butterfield | 否 |
| 45 | `Authors/Cavell, Stanley/In Quest of the Ordinary_ Lines - Stanley Cavell.pdf` | In Quest of the Ordinary_ Lines - Stanley Cavell.pdf | In Quest of the Ordinary_ Lines | Stanley Cavell | 否 |
| 46 | `Authors/Cavell, Stanley/The Claim of Reason_ Wittgenste - Stanley Cavell.pdf` | The Claim of Reason_ Wittgenste - Stanley Cavell.pdf | The Claim of Reason_ Wittgenste | Stanley Cavell | 否 |
| 47 | `Authors/Chun, Wendy Hui Kyong/Control and Freedom_ Power and - Wendy Hui Kyong Chun.pdf` | Control and Freedom_ Power and - Wendy Hui Kyong Chun.pdf | Control and Freedom_ Power and | Wendy Hui Kyong Chun | 否 |
| 48 | `Authors/Culler, Jonathan/Culler, Jonathan D. - Theory of the lyric-Harvard University Press (2017).pdf` | Culler, Jonathan D. - Theory of the lyric-Harvard University Press (2017).pdf | Culler, Jonathan D | Jonathan Culler | 否 |
| 49 | `Authors/Culler, Jonathan/文学理论入门_英汉对照_by_美_乔纳森卡勒_李平_z_lib_org.pdf` | 文学理论入门_英汉对照_by_美_乔纳森卡勒_李平_z_lib_org.pdf | 文学理论入门_英汉对照_by_美_乔纳森卡勒_李平_z_lib_org | Jonathan Culler | 否 |
| 50 | `Authors/Danto, Arthur C/Arthur C. Danto The Philosophical Disenfranchisement of Art - 未知.pdf` | Arthur C. Danto The Philosophical Disenfranchisement of Art - 未知.pdf | The Philosophical Disenfranchisement of Art | Arthur C. Danto | 否 |
| 51 | `Authors/Danto, Arthur C/Arthur C. Danto The Transfiguration of the Commonplace A Philosophy of Art - 未知.pdf` | Arthur C. Danto The Transfiguration of the Commonplace A Philosophy of Art - 未知.pdf | The Transfiguration of the Commonplace A Philosophy of Art | Arthur C. Danto | 否 |
| 52 | `Authors/Deleuze, Gilles/Joe Hughes - Deleuze's 'Difference and Repetition'_ A Reader's Guide (Reader's Guides) (2009, Co` | Joe Hughes - Deleuze's 'Difference and Repetition'_ A Reader's Guide (Reader's Guides) (2009, Continuum).pdf | Deleuze's 'Difference and Repetition'_ A Reader's Guide | Gilles Deleuze\|Joe Hughes | 否 |
| 53 | `Authors/Deleuze, Gilles/【德勒兹】弗朗西斯·培根：感觉的逻辑.pdf` | 【德勒兹】弗朗西斯·培根：感觉的逻辑.pdf | 【德勒兹】弗朗西斯·培根：感觉的逻辑 | Gilles Deleuze | 否 |
| 54 | `Authors/Deleuze, Gilles/【德勒兹】普鲁斯特和符号（法文）.pdf` | 【德勒兹】普鲁斯特和符号（法文）.pdf | 【德勒兹】普鲁斯特和符号（法文） | Gilles Deleuze | 否 |
| 55 | `Authors/Deleuze, Gilles/【德勒兹】荒岛及其他文本.pdf` | 【德勒兹】荒岛及其他文本.pdf | 【德勒兹】荒岛及其他文本 | Gilles Deleuze | 否 |
| 56 | `Authors/Deleuze, Gilles/【德勒兹】電影1：運動一影像.pdf` | 【德勒兹】電影1：運動一影像.pdf | 【德勒兹】電影1：運動一影像 | Gilles Deleuze | 否 |
| 57 | `Authors/Dennett, Daniel/Brainchildren_ Essays on Designing Minds_Daniel C. Dennett.pdf` | Brainchildren_ Essays on Designing Minds_Daniel C. Dennett.pdf | Brainchildren_ Essays on Designing Minds | Daniel Dennett | 否 |
| 58 | `Authors/Deren, Maya/Bill Nichols - Maya Deren and the American Avant-Garde-University of California Press (2001).pdf` | Bill Nichols - Maya Deren and the American Avant-Garde-University of California Press (2001).pdf | Maya Deren and the American Avant-Garde | Maya Deren\|Bill Nichols | 否 |
| 59 | `Authors/Deren, Maya/Maya Deren - Divine Horsemen_ The Living Gods of Haiti-McPherson (1983).pdf` | Maya Deren - Divine Horsemen_ The Living Gods of Haiti-McPherson (1983).pdf | Divine Horsemen_ The Living Gods of Haiti | Maya Deren | 否 |
| 60 | `Authors/Deren, Maya/Maya Deren - Divine Horsemen_ The Living Gods of Haiti-Thames and Hudson (1953).pdf` | Maya Deren - Divine Horsemen_ The Living Gods of Haiti-Thames and Hudson (1953).pdf | Divine Horsemen_ The Living Gods of Haiti | Maya Deren | 否 |
| 61 | `Authors/Derrida, Jacques/Jacques Derrida - The Truth in Painting (1987).pdf` | Jacques Derrida - The Truth in Painting (1987).pdf | The Truth in Painting | Jacques Derrida | 否 |
| 62 | `Authors/Dickie, George/Art and Value Dickie.pdf` | Art and Value Dickie.pdf | Art and Value Dickie | George Dickie | 否 |
| 63 | `Authors/Dilthey, Wilhelm/(Galaxy Books) R. G. Collingwood - Idea of History-OUP Australia and New Zealand (1956).pdf` | (Galaxy Books) R. G. Collingwood - Idea of History-OUP Australia and New Zealand (1956).pdf | Idea of History-OUP Australia and New Zealand | Wilhelm Dilthey\|R. G. Collingwood | 否 |
| 64 | `Authors/Dufrenne, Mikel/美学与哲学（杜夫海纳）.pdf` | 美学与哲学（杜夫海纳）.pdf | 美学与哲学（杜夫海纳） | Mikel Dufrenne | 否 |
| 65 | `Authors/Durkheim, Emile/Emile Durkheim, Neil Gross, Robert Alun Jones, Hans Joas - Durkheim's Philosophy Lectures_ Notes` | Emile Durkheim, Neil Gross, Robert Alun Jones, Hans Joas - Durkheim's Philosophy Lectures_ Notes from the Lycée de Sens  | Durkheim's Philosophy Lectures_ Notes from the Lycée de Sens Course, 1883-1884 | Émile Durkheim | 否 |
| 66 | `Authors/Durkheim, Emile/Émile Durkheim - Socialism and Saint Simon-Taylor and Francis (2009).pdf` | Émile Durkheim - Socialism and Saint Simon-Taylor and Francis (2009).pdf | Socialism and Saint Simon-Taylor and Francis | Émile Durkheim | 否 |
| 67 | `Authors/Dyer-Witheford, Nick/Stephen Kline, Nick Dyer-Witheford, Greig de Peuter - Digital play_ the interaction of tech` | Stephen Kline, Nick Dyer-Witheford, Greig de Peuter - Digital play_ the interaction of technology, culture, and marketin | Digital play_ the interaction of technology, culture, and marketing | Nick Dyer-Witheford | 否 |
| 68 | `Authors/Eagleton, Terry/美学意识形态_ （修订版）_[英] 特里·伊格尔顿(Terry Eagleton).pdf` | 美学意识形态_ （修订版）_[英] 特里·伊格尔顿(Terry Eagleton).pdf | 美学意识形态_ （修订版） | Terry Eagleton | 否 |
| 69 | `Authors/Eco, Umberto/Umberto Eco - The Open Work-Harvard University Press (1989).pdf` | Umberto Eco - The Open Work-Harvard University Press (1989).pdf | The Open Work | Umberto Eco | 否 |
| 70 | `Authors/Elkins, James/绘画与眼泪_詹姆斯·埃尔金斯.pdf` | 绘画与眼泪_詹姆斯·埃尔金斯.pdf | 绘画与眼泪 | James Elkins | 否 |
| 71 | `Authors/Elkins, James/绘画是什么？如何用炼金术语言思考油画 -- 詹姆斯·埃尔金斯 -- 2021 -- 重庆大学出版社 -- 9787568926607 -- 34b91dfbb0ae3e97934da47536a1` | 绘画是什么？如何用炼金术语言思考油画 -- 詹姆斯·埃尔金斯 -- 2021 -- 重庆大学出版社 -- 9787568926607 -- 34b91dfbb0ae3e97934da47536a1fb75 -- Anna’s Archive | 绘画是什么？如何用炼金术语言思考油画 -- 詹姆斯·埃尔金斯 -- 2021 -- 重庆大学出版社 | James Elkins | 否 |
| 72 | `Authors/Elkins, James/西方艺术史中的中国山水画 -- [美]詹姆斯·埃尔金斯（James Elkins） -- 9787547917404 -- 51bb952fd13c9ba3afc80498c9fb1bb5 -- ` | 西方艺术史中的中国山水画 -- [美]詹姆斯·埃尔金斯（James Elkins） -- 9787547917404 -- 51bb952fd13c9ba3afc80498c9fb1bb5 -- Anna’s Archive.pdf | 西方艺术史中的中国山水画 | James Elkins | 否 |
| 73 | `Authors/Fabian, Johannes/Time and the Other How Anthropology Makes Its Object-Columbia University Press (1983) - Fabian ` | Time and the Other How Anthropology Makes Its Object-Columbia University Press (1983) - Fabian Johannes.pdf | Time and the Other How Anthropology Makes Its Object | Fabian Johannes | 否 |
| 74 | `Authors/Feenberg, Andrew/Between Reason and Experience Essays in Technology and Modernity-The MIT Press (2010).pdf` | Between Reason and Experience Essays in Technology and Modernity-The MIT Press (2010).pdf | Between Reason and Experience Essays in Technology and Modernity | Andrew Feenberg | 否 |
| 75 | `Authors/Feenberg, Andrew/可选择的现代性 芬伯格.pdf` | 可选择的现代性 芬伯格.pdf | 可选择的现代性 芬伯格 | Andrew Feenberg | 否 |
| 76 | `Authors/Feenberg, Andrew/技术体系 理性的社会生活 （加）安德鲁·芬伯格著；上海社会科学院科学技术哲学创新团队译.pdf` | 技术体系 理性的社会生活 （加）安德鲁·芬伯格著；上海社会科学院科学技术哲学创新团队译.pdf | 技术体系 理性的社会生活 （加）安德鲁·芬伯格著；上海社会科学院科学技术哲学创新团队译 | Andrew Feenberg | 否 |
| 77 | `Authors/Feenberg, Andrew/技术批判理论 （美）安德鲁·芬伯格（Andrew Feenberg）著；韩连庆，曹观法译.pdf` | 技术批判理论 （美）安德鲁·芬伯格（Andrew Feenberg）著；韩连庆，曹观法译.pdf | 技术批判理论 | Andrew Feenberg | 否 |
| 78 | `Authors/Fiske, John/Reading television  _John Fiske, John Hartley.pdf` | Reading television  _John Fiske, John Hartley.pdf | Reading television | John Fiske | 否 |
| 79 | `Authors/Fiske, John/Television Culture_ Popular Pleasures and Politics (Studies in Communication Series)_John Fiske.pdf` | Television Culture_ Popular Pleasures and Politics (Studies in Communication Series)_John Fiske.pdf | Television Culture_ Popular Pleasures and Politics | John Fiske | 否 |
| 80 | `Authors/Floridi, Luciano/The Ethics of Information_Luciano Floridi.pdf` | The Ethics of Information_Luciano Floridi.pdf | The Ethics of Information | Luciano Floridi | 否 |
| 81 | `Authors/Floridi, Luciano/The Fourth Revolution_ How the infosphere is reshaping human reality_Luciano Floridi.pdf` | The Fourth Revolution_ How the infosphere is reshaping human reality_Luciano Floridi.pdf | The Fourth Revolution_ How the infosphere is reshaping human reality | Luciano Floridi | 否 |
| 82 | `Authors/Floridi, Luciano/The Logic Of Information_ A Theory Of Philosophy As Conceptual Design_Luciano Floridi.pdf` | The Logic Of Information_ A Theory Of Philosophy As Conceptual Design_Luciano Floridi.pdf | The Logic Of Information_ A Theory Of Philosophy As Conceptual Design | Luciano Floridi | 否 |
| 83 | `Authors/Floridi, Luciano/The Philosophy of Information Quality_Luciano Floridi, Phyllis Illari (eds.).pdf` | The Philosophy of Information Quality_Luciano Floridi, Phyllis Illari (eds.).pdf | The Philosophy of Information Quality | Luciano Floridi | 否 |
| 84 | `Authors/Flusser, Vilem/Flusser, Vilém - Philosophy of Language (2016) - libgen.li.pdf` | Flusser, Vilém - Philosophy of Language (2016) - libgen.li.pdf | Philosophy of Language | Vilém Flusser | 否 |
| 85 | `Authors/Flusser, Vilem/METAFLUXUS.pdf` | METAFLUXUS.pdf | METAFLUXUS | Vilem Flusser | 否 |
| 86 | `Authors/Flusser, Vilem/Vilem Flusser - Into the Universe of Technical Images   (2011, Univ Of Minnesota Press) - libgen.` | Vilem Flusser - Into the Universe of Technical Images   (2011, Univ Of Minnesota Press) - libgen.lc.pdf | Into the Universe of Technical Images | Vilém Flusser | 否 |
| 87 | `Authors/Flusser, Vilem/Vilem Flusser - Kommunikologie.pdf` | Vilem Flusser - Kommunikologie.pdf | Kommunikologie | Vilém Flusser | 否 |
| 88 | `Authors/Flusser, Vilem/Vilem Flusser - Medienkultur.pdf` | Vilem Flusser - Medienkultur.pdf | Medienkultur | Vilém Flusser | 否 |
| 89 | `Authors/Flusser, Vilem/Vilém Flusser - Für eine Philosophie der Fotografie-European Photgraphy (2011).pdf` | Vilém Flusser - Für eine Philosophie der Fotografie-European Photgraphy (2011).pdf | Für eine Philosophie der Fotografie-European Photgraphy | Vilém Flusser | 否 |
| 90 | `Authors/Flusser, Vilem/Vilém Flusser - Lob der Oberflächlichkeit. Für eine Phänomenologie der Medien (1993, Bollmann) - ` | Vilém Flusser - Lob der Oberflächlichkeit. Für eine Phänomenologie der Medien (1993, Bollmann) - libgen.lc.pdf | Lob der Oberflächlichkeit. Für eine Phänomenologie der Medien (1993, Bollmann) | Vilém Flusser | 否 |
| 91 | `Authors/Flusser, Vilem/[Metaflux __ Vilém Flusser] Vilem Flusser - The Surprising Phenomenon of Human Communication (20` | [Metaflux __ Vilém Flusser] Vilem Flusser - The Surprising Phenomenon of Human Communication (2016, Metaflux Publishing | The Surprising Phenomenon of Human Communication | Vilém Flusser | 否 |
| 92 | `Authors/Flusser, Vilem/技术图像的宇宙 弗鲁塞尔.pdf` | 技术图像的宇宙 弗鲁塞尔.pdf | 技术图像的宇宙 弗鲁塞尔 | Vilem Flusser | 否 |
| 93 | `Authors/Flusser, Vilem/摄影哲学的思考_威廉_弗卢塞尔,Vilém_Flusser_z_lib_org.pdf` | 摄影哲学的思考_威廉_弗卢塞尔,Vilém_Flusser_z_lib_org.pdf | z_lib_org | Vilem Flusser | 否 |
| 94 | `Authors/Fodor, Jerry/心理模块性 (（美）J.A.福多（Jerry A.Fodor）著；李丽译 etc.) (Z-Library).pdf` | 心理模块性 (（美）J.A.福多（Jerry A.Fodor）著；李丽译 etc.) (Z-Library).pdf | 心理模块性 | Jerry Fodor | 否 |
| 95 | `Authors/Foster, Hal/Anti-aesthetic Essays on postmodern culture-Bay Press (1983), The.pdf` | Anti-aesthetic Essays on postmodern culture-Bay Press (1983), The.pdf | Anti-aesthetic Essays on postmodern culture | Hal Foster | 否 |
| 96 | `Authors/Foster, Hal/Return of the Real The Avant-Garde at the End of the Century-The MIT Press (1996), The.pdf` | Return of the Real The Avant-Garde at the End of the Century-The MIT Press (1996), The.pdf | Return of the Real The Avant-Garde at the End of the Century | Hal Foster | 否 |
| 97 | `Authors/Foucault, Michel/对活人的治理 福柯.pdf` | 对活人的治理 福柯.pdf | 对活人的治理 福柯 | Michel Foucault | 否 |
| 98 | `Authors/Foucault, Michel/知识意志讲稿 福柯.pdf` | 知识意志讲稿 福柯.pdf | 知识意志讲稿 福柯 | Michel Foucault | 否 |
| 99 | `Authors/Franke, Herbert W/Franke, Herbert W - Der Orchideenkäfig.pdf` | Franke, Herbert W - Der Orchideenkäfig.pdf | Der Orchideenkäfig | Herbert W Franke | 否 |
| 100 | `Authors/Friedrich Schiller/审美教育书简[德]席勒.冯至、范大灿译.北京大学出版社(1985).pdf` | 审美教育书简[德]席勒.冯至、范大灿译.北京大学出版社(1985).pdf | 审美教育书简 | Friedrich Schiller | 是 |
| 101 | `Authors/Galloway, Alexander R/Alexander R. Galloway - Protocol_ How Control Exists after Decentralization (Leonardo Book` | Alexander R. Galloway - Protocol_ How Control Exists after Decentralization (Leonardo Books) (2004).pdf | Protocol_ How Control Exists after Decentralization | Alexander R. Galloway | 否 |
| 102 | `Authors/Galloway, Alexander R/Alexander R. Galloway - The Interface Effect-Polity (2012).pdf` | Alexander R. Galloway - The Interface Effect-Polity (2012).pdf | The Interface Effect | Alexander R. Galloway | 否 |
| 103 | `Authors/Geertz, Clifford/Geertz Clifford The Interpretation of Cultures Selected Essays.pdf` | Geertz Clifford The Interpretation of Cultures Selected Essays.pdf | The Interpretation of Cultures Selected Essays | Clifford Geertz | 否 |
| 104 | `Authors/Gombrich, E H/象征的图像.pdf` | 象征的图像.pdf | 象征的图像 | E H Gombrich | 否 |
| 105 | `Authors/Goodman, Nelson/Springer) Nelson Goodman - The Structure of Appearance (1977.pdf` | Springer) Nelson Goodman - The Structure of Appearance (1977.pdf | The Structure of Appearance | Nelson Goodman | 否 |
| 106 | `Authors/Graeber, David/Toward An Anthropological Theor - David Graeber.pdf` | Toward An Anthropological Theor - David Graeber.pdf | Toward An Anthropological Theor | David Graeber | 否 |
| 107 | `Authors/Groys, Boris/[Columbia Themes in Philosophy, Social Criticism, and the Arts] Boris Groys, Carsten Strathausen - ` | [Columbia Themes in Philosophy, Social Criticism, and the Arts] Boris Groys, Carsten Strathausen - Under Suspicion_ A Ph | Carsten Strathausen - Under Suspicion_ A Phenomenology of Media | Boris Groys | 否 |
| 108 | `Authors/Habermas, Jurgen/(社会理论译丛) （德）尤尔根・哈贝马斯 - 重建历史唯物主义-社会科学文献出版社 (2000).pdf` | (社会理论译丛) （德）尤尔根・哈贝马斯 - 重建历史唯物主义-社会科学文献出版社 (2000).pdf | 重建历史唯物主义-社会科学文献出版社 | Jürgen Habermas | 否 |
| 109 | `Authors/Habermas, Jurgen/Karl Jaspers - General Psychopathology-University of Chicago Press, University of Toronto Press` | Karl Jaspers - General Psychopathology-University of Chicago Press, University of Toronto Press (1963).pdf | General Psychopathology | Jurgen Habermas\|Karl Jaspers | 否 |
| 110 | `Authors/Handke, Peter/Handbook on the Digital Creative Economy_Ruth Towse (editor), Christian Handke (editor).pdf` | Handbook on the Digital Creative Economy_Ruth Towse (editor), Christian Handke (editor).pdf | Handbook on the Digital Creative Economy_Ruth Towse | Peter Handke | 否 |
| 111 | `Authors/Hardt, Michael/Assembly_Michael Hardt, Antonio Negri.pdf` | Assembly_Michael Hardt, Antonio Negri.pdf | Assembly | Michael Hardt | 否 |
| 112 | `Authors/Hardt, Michael/Labor of Dionysus_ A Critique of the State-Form  _Michael Hardt & Antonio Negri.pdf` | Labor of Dionysus_ A Critique of the State-Form  _Michael Hardt & Antonio Negri.pdf | Labor of Dionysus_ A Critique of the State-Form | Michael Hardt | 否 |
| 113 | `Authors/Hardt, Michael/Multitude_ War and Democracy in the Age of Empire  _Michael Hardt, Antonio Negri.pdf` | Multitude_ War and Democracy in the Age of Empire  _Michael Hardt, Antonio Negri.pdf | Multitude_ War and Democracy in the Age of Empire | Michael Hardt | 否 |
| 114 | `Authors/Hardt, Michael/Radical Thought in Italy_ A Potential Politics_Paolo Virno, Michael Hardt.pdf` | Radical Thought in Italy_ A Potential Politics_Paolo Virno, Michael Hardt.pdf | Radical Thought in Italy_ A Potential Politics | Michael Hardt | 否 |
| 115 | `Authors/Harman, Graham/immaterialism-objects-and-social-theory-graham-1--annas-archive--libgenrs-nf-2225539.pdf` | immaterialism-objects-and-social-theory-graham-1--annas-archive--libgenrs-nf-2225539.pdf | immaterialism-objects-and-social-theory | Graham Harman | 否 |
| 116 | `Authors/Hartmann, Nicolai/Media Events_ The Live Broadcasting of History_Daniel Dayan, Elihu Katz.pdf` | Media Events_ The Live Broadcasting of History_Daniel Dayan, Elihu Katz.pdf | Media Events_ The Live Broadcasting of History_Daniel Dayan, Elihu Katz | Nicolai Hartmann | 否 |
| 117 | `Authors/Harvey, David/大卫·哈维_ David Harvey - 巴黎城记：现代性之都的诞生-广西师范大学出版社 (2010).pdf` | 大卫·哈维_ David Harvey - 巴黎城记：现代性之都的诞生-广西师范大学出版社 (2010).pdf | 巴黎城记：现代性之都的诞生-广西师范大学出版社 | David Harvey | 否 |
| 118 | `Authors/Harvey, David/大卫·哈维_ David Harvey - 资本之谜：人人需要知道的资本主义真相-电子工业出版社 (2011).pdf` | 大卫·哈维_ David Harvey - 资本之谜：人人需要知道的资本主义真相-电子工业出版社 (2011).pdf | 资本之谜：人人需要知道的资本主义真相-电子工业出版社 | David Harvey | 否 |
| 119 | `Authors/Hayles, N Katherine/N. Katherine Hayles - Postprint_ Books and Becoming Computational-Columbia University Press ` | N. Katherine Hayles - Postprint_ Books and Becoming Computational-Columbia University Press (2021).pdf | Postprint_ Books and Becoming Computational | N. Katherine Hayles | 否 |
| 120 | `Authors/Hayles, N Katherine/N. Katherine Hayles - Unthought_ The Power of the Cognitive Nonconscious-University of Chica` | N. Katherine Hayles - Unthought_ The Power of the Cognitive Nonconscious-University of Chicago Press (2017).pdf | Unthought_ The Power of the Cognitive Nonconscious | N. Katherine Hayles | 否 |
| 121 | `Authors/Hegel, GWF/Hegel - Werke - Band 11 - Berliner Schriften 1818-1831 Hegel.pdf` | Hegel - Werke - Band 11 - Berliner Schriften 1818-1831 Hegel.pdf | Werke - Band 11 - Berliner Schriften 1818 | Hegel | 否 |
| 122 | `Authors/Hegel, GWF/Hegel - Werke - Band 20 - Vorlesungen über die Geschichte der Philosophie III - OCRed.pdf` | Hegel - Werke - Band 20 - Vorlesungen über die Geschichte der Philosophie III - OCRed.pdf | Werke - Band 20 - Vorlesungen über die Geschichte der Philosophie III | GWF Hegel | 否 |
| 123 | `Authors/Heidegger, Martin/(海德格尔文集) [德] 马丁·海德格尔 - 哲学论稿（从本有而来）-商务印书馆 (2014).pdf` | (海德格尔文集) [德] 马丁·海德格尔 - 哲学论稿（从本有而来）-商务印书馆 (2014).pdf | 哲学论稿（从本有而来）-商务印书馆 | Martin Heidegger | 否 |
| 124 | `Authors/Heidegger, Martin/(海德格尔文集) [德] 马丁·海德格尔 - 时间概念史导论-商务印书馆 (2014).pdf` | (海德格尔文集) [德] 马丁·海德格尔 - 时间概念史导论-商务印书馆 (2014).pdf | 时间概念史导论-商务印书馆 | Martin Heidegger | 否 |
| 125 | `Authors/Heidegger, Martin/(海德格尔文集) [德]马丁·海德格尔 - 在通向语言的途中-商务印书馆 (2015).pdf` | (海德格尔文集) [德]马丁·海德格尔 - 在通向语言的途中-商务印书馆 (2015).pdf | 在通向语言的途中-商务印书馆 | Martin Heidegger | 否 |
| 126 | `Authors/Heidegger, Martin/(海德格尔文集) [德]马丁·海德格尔 - 论哲学的规定-商务印书馆 (2015).pdf` | (海德格尔文集) [德]马丁·海德格尔 - 论哲学的规定-商务印书馆 (2015).pdf | 论哲学的规定-商务印书馆 | Martin Heidegger | 否 |
| 127 | `Authors/Heidegger, Martin/(海德格尔文集) 海德格尔 - 柏拉图的智者-商务印书馆 (2015).pdf` | (海德格尔文集) 海德格尔 - 柏拉图的智者-商务印书馆 (2015).pdf | 柏拉图的智者-商务印书馆 | Martin Heidegger | 否 |
| 128 | `Authors/Heidegger, Martin/(海德格尔文集) 马丁·海德格尔 - 不莱梅和弗莱堡演讲-商务印书馆 (2018).pdf` | (海德格尔文集) 马丁·海德格尔 - 不莱梅和弗莱堡演讲-商务印书馆 (2018).pdf | 不莱梅和弗莱堡演讲-商务印书馆 | Martin Heidegger | 否 |
| 129 | `Authors/Heidegger, Martin/(海德格尔文集) 马丁·海德格尔 - 同一与差异-商务印书馆 (2014).pdf` | (海德格尔文集) 马丁·海德格尔 - 同一与差异-商务印书馆 (2014).pdf | 同一与差异-商务印书馆 | Martin Heidegger | 否 |
| 130 | `Authors/Heidegger, Martin/Heidegger - Grundbegriffe der aristotelischen Philosophie.pdf` | Heidegger - Grundbegriffe der aristotelischen Philosophie.pdf | Grundbegriffe der aristotelischen Philosophie | Martin Heidegger | 否 |
| 131 | `Authors/Heidegger, Martin/Martin Heidegger - Aus der Erfahrung des Denkens.pdf` | Martin Heidegger - Aus der Erfahrung des Denkens.pdf | Aus der Erfahrung des Denkens | Martin Heidegger | 否 |
| 132 | `Authors/Heidegger, Martin/Martin Heidegger - Beiträge zur Philosophie (Vom Ereignis) (1936-1938).pdf` | Martin Heidegger - Beiträge zur Philosophie (Vom Ereignis) (1936-1938).pdf | Beiträge zur Philosophie | Martin Heidegger | 否 |
| 133 | `Authors/Heidegger, Martin/Martin Heidegger - Bremer und Freiburger Vorträge.pdf` | Martin Heidegger - Bremer und Freiburger Vorträge.pdf | Bremer und Freiburger Vorträge | Martin Heidegger | 否 |
| 134 | `Authors/Heidegger, Martin/Martin Heidegger - Das Ereignis (1941_42).pdf` | Martin Heidegger - Das Ereignis (1941_42).pdf | Das Ereignis (1941_42) | Martin Heidegger | 否 |
| 135 | `Authors/Heidegger, Martin/Martin Heidegger - Der Begriff der Zeit (1924).pdf` | Martin Heidegger - Der Begriff der Zeit (1924).pdf | Der Begriff der Zeit | Martin Heidegger | 否 |
| 136 | `Authors/Heidegger, Martin/Martin Heidegger - Der Deutsche Idealismus.pdf` | Martin Heidegger - Der Deutsche Idealismus.pdf | Der Deutsche Idealismus | Martin Heidegger | 否 |
| 137 | `Authors/Heidegger, Martin/Martin Heidegger - Der Satz vom Grund (1955-1956).pdf` | Martin Heidegger - Der Satz vom Grund (1955-1956).pdf | Der Satz vom Grund (1955-1956) | Martin Heidegger | 否 |
| 138 | `Authors/Heidegger, Martin/Martin Heidegger - Die Frage nach dem Ding. Zu Kants Lehre von den transzendentalen Grundsätze` | Martin Heidegger - Die Frage nach dem Ding. Zu Kants Lehre von den transzendentalen Grundsätzen.pdf | Die Frage nach dem Ding. Zu Kants Lehre von den transzendentalen Grundsätzen | Martin Heidegger | 否 |
| 139 | `Authors/Heidegger, Martin/Martin Heidegger - Die Geschichte des Seyns.pdf` | Martin Heidegger - Die Geschichte des Seyns.pdf | Die Geschichte des Seyns | Martin Heidegger | 否 |
| 140 | `Authors/Heidegger, Martin/Martin Heidegger - Die Grundbegriffe der Metaphysik_ Welt - Endlichkeit - Einsamkeit (Winterse` | Martin Heidegger - Die Grundbegriffe der Metaphysik_ Welt - Endlichkeit - Einsamkeit (Wintersemester 1929_30).pdf | Die Grundbegriffe der Metaphysik_ Welt - Endlichkeit - Einsamkeit | Martin Heidegger | 否 |
| 141 | `Authors/Heidegger, Martin/Martin Heidegger - Die Grundprobleme der Phänomenologie (Sommersemester 1927).pdf` | Martin Heidegger - Die Grundprobleme der Phänomenologie (Sommersemester 1927).pdf | Die Grundprobleme der Phänomenologie | Martin Heidegger | 否 |
| 142 | `Authors/Heidegger, Martin/Martin Heidegger - Einführung in die Metaphysik (Sommersemester 1935).pdf` | Martin Heidegger - Einführung in die Metaphysik (Sommersemester 1935).pdf | Einführung in die Metaphysik | Martin Heidegger | 否 |
| 143 | `Authors/Heidegger, Martin/Martin Heidegger - Einleitung in die Philosophie (Wintersemester 1928–29).pdf` | Martin Heidegger - Einleitung in die Philosophie (Wintersemester 1928–29).pdf | Einleitung in die Philosophie | Martin Heidegger | 否 |
| 144 | `Authors/Heidegger, Martin/Martin Heidegger - Erläuterungen zu Hölderlins Dichtung.pdf` | Martin Heidegger - Erläuterungen zu Hölderlins Dichtung.pdf | Erläuterungen zu Hölderlins Dichtung | Martin Heidegger | 否 |
| 145 | `Authors/Heidegger, Martin/Martin Heidegger - Feldweg-Gespräche (1944_45).pdf` | Martin Heidegger - Feldweg-Gespräche (1944_45).pdf | Feldweg-Gespräche (1944_45) | Martin Heidegger | 否 |
| 146 | `Authors/Heidegger, Martin/Martin Heidegger - Frühe Schriften (1912-1916).pdf` | Martin Heidegger - Frühe Schriften (1912-1916).pdf | Frühe Schriften (1912-1916) | Martin Heidegger | 否 |
| 147 | `Authors/Heidegger, Martin/Martin Heidegger - Gedachtes.pdf` | Martin Heidegger - Gedachtes.pdf | Gedachtes | Martin Heidegger | 否 |
| 148 | `Authors/Heidegger, Martin/Martin Heidegger - Grundbegriffe der antiken Philosophie.pdf` | Martin Heidegger - Grundbegriffe der antiken Philosophie.pdf | Grundbegriffe der antiken Philosophie | Martin Heidegger | 否 |
| 149 | `Authors/Heidegger, Martin/Martin Heidegger - Grundprobleme der Phänomenologie (Wintersemester 1919-20).pdf` | Martin Heidegger - Grundprobleme der Phänomenologie (Wintersemester 1919-20).pdf | Grundprobleme der Phänomenologie | Martin Heidegger | 否 |
| 150 | `Authors/Heidegger, Martin/Martin Heidegger - Hegel_ 1. Die Negativität (1938_39), 2. Erläuterungen der __Einleitung__ zu` | Martin Heidegger - Hegel_ 1. Die Negativität (1938_39), 2. Erläuterungen der __Einleitung__ zu Hegels __Phänomenologie d | Hegel_ 1. Die Negativität (1938_39), 2. Erläuterungen der __Einleitung__ zu Hegels __Phänomenologie des Geistes | Martin Heidegger | 否 |
| 151 | `Authors/Heidegger, Martin/Martin Heidegger - Hegels Phänomenologie des Geistes (Wintersemester 1930_31).pdf` | Martin Heidegger - Hegels Phänomenologie des Geistes (Wintersemester 1930_31).pdf | Hegels Phänomenologie des Geistes | Martin Heidegger | 否 |
| 152 | `Authors/Heidegger, Martin/Martin Heidegger - Holzwege (1935–1946).pdf` | Martin Heidegger - Holzwege (1935–1946).pdf | Holzwege (1935–1946) | Martin Heidegger | 否 |
| 153 | `Authors/Heidegger, Martin/Martin Heidegger - Hölderlins Hymne __Der Ister__ (Sommersemester 1942).pdf` | Martin Heidegger - Hölderlins Hymne __Der Ister__ (Sommersemester 1942).pdf | Hölderlins Hymne __Der Ister | Martin Heidegger | 否 |
| 154 | `Authors/Heidegger, Martin/Martin Heidegger - Hölderlins Hymnen 'Germanien' und 'Der Rhein'.pdf` | Martin Heidegger - Hölderlins Hymnen 'Germanien' und 'Der Rhein'.pdf | Hölderlins Hymnen 'Germanien' und 'Der Rhein' | Martin Heidegger | 否 |
| 155 | `Authors/Heidegger, Martin/Martin Heidegger - Identität und Differenz (1955-1957).pdf` | Martin Heidegger - Identität und Differenz (1955-1957).pdf | Identität und Differenz (1955-1957) | Martin Heidegger | 否 |
| 156 | `Authors/Heidegger, Martin/Martin Heidegger - Kant und das Problem der Metaphysik (1929).pdf` | Martin Heidegger - Kant und das Problem der Metaphysik (1929).pdf | Kant und das Problem der Metaphysik | Martin Heidegger | 否 |
| 157 | `Authors/Heidegger, Martin/Martin Heidegger - Leitgedanken zur Entstehung der Metaphysik, der neuzeitlichen Wissenschaft ` | Martin Heidegger - Leitgedanken zur Entstehung der Metaphysik, der neuzeitlichen Wissenschaft und der modernen Technik.p | Leitgedanken zur Entstehung der Metaphysik, der neuzeitlichen Wissenschaft und der modernen Technik | Martin Heidegger | 否 |
| 158 | `Authors/Heidegger, Martin/Martin Heidegger - Metaphysik und Nihilismus 1. Die Uberwindung der Metaphysik (1938-39) 2. Da` | Martin Heidegger - Metaphysik und Nihilismus 1. Die Uberwindung der Metaphysik (1938-39) 2. Das Wesen des Nihilismus (19 | Metaphysik und Nihilismus 1. Die Uberwindung der Metaphysik (1938-39) 2. Das Wesen des Nihilismus (1946-1948) | Martin Heidegger | 否 |
| 159 | `Authors/Heidegger, Martin/Martin Heidegger - Metaphysische Anfangsgründe der Logik im Ausgang von Leibniz (Sommersemeste` | Martin Heidegger - Metaphysische Anfangsgründe der Logik im Ausgang von Leibniz (Sommersemester 1928).pdf | Metaphysische Anfangsgründe der Logik im Ausgang von Leibniz | Martin Heidegger | 否 |
| 160 | `Authors/Heidegger, Martin/Martin Heidegger - Nietzsche I.pdf` | Martin Heidegger - Nietzsche I.pdf | Nietzsche I | Martin Heidegger | 否 |
| 161 | `Authors/Heidegger, Martin/Martin Heidegger - Nietzsche II (1939-1946).pdf` | Martin Heidegger - Nietzsche II (1939-1946).pdf | Nietzsche II (1939-1946) | Martin Heidegger | 否 |
| 162 | `Authors/Heidegger, Martin/Martin Heidegger - Nietzsches Lehre vom Willen zur Macht als Erkenntnis.pdf` | Martin Heidegger - Nietzsches Lehre vom Willen zur Macht als Erkenntnis.pdf | Nietzsches Lehre vom Willen zur Macht als Erkenntnis | Martin Heidegger | 否 |
| 163 | `Authors/Heidegger, Martin/Martin Heidegger - Nietzsches Metaphysik - Einleitung in die Philosophie - Denken und Dichten.` | Martin Heidegger - Nietzsches Metaphysik - Einleitung in die Philosophie - Denken und Dichten.pdf | Nietzsches Metaphysik - Einleitung in die Philosophie | Martin Heidegger | 否 |
| 164 | `Authors/Heidegger, Martin/Martin Heidegger - Ontologie. Hermeneutik der Faktizität.pdf` | Martin Heidegger - Ontologie. Hermeneutik der Faktizität.pdf | Ontologie. Hermeneutik der Faktizität | Martin Heidegger | 否 |
| 165 | `Authors/Heidegger, Martin/Martin Heidegger - Parmenides (Wintersemester 1942_43).pdf` | Martin Heidegger - Parmenides (Wintersemester 1942_43).pdf | Parmenides | Martin Heidegger | 否 |
| 166 | `Authors/Heidegger, Martin/Martin Heidegger - Phänomenologie der Anschauung und des Ausdrucks. Theorie der philosophische` | Martin Heidegger - Phänomenologie der Anschauung und des Ausdrucks. Theorie der philosophischen Begriffsbildung.pdf | Phänomenologie der Anschauung und des Ausdrucks. Theorie der philosophischen Begriffsbildung | Martin Heidegger | 否 |
| 167 | `Authors/Heidegger, Martin/Martin Heidegger - Phänomenologische Interpretation von Kants Kritik der reinen Vernunft (Wint` | Martin Heidegger - Phänomenologische Interpretation von Kants Kritik der reinen Vernunft (Wintersemester 1927–28).pdf | Phänomenologische Interpretation von Kants Kritik der reinen Vernunft | Martin Heidegger | 否 |
| 168 | `Authors/Heidegger, Martin/Martin Heidegger - Phänomenologische Interpretationen ausgewählter Abhandlungen des Aristotele` | Martin Heidegger - Phänomenologische Interpretationen ausgewählter Abhandlungen des Aristoteles zur Ontologie und Logik. | Phänomenologische Interpretationen ausgewählter Abhandlungen des Aristoteles zur Ontologie und Logik | Martin Heidegger | 否 |
| 169 | `Authors/Heidegger, Martin/Martin Heidegger - Platon_ Sophistes (Wintersemester 1924-25).pdf` | Martin Heidegger - Platon_ Sophistes (Wintersemester 1924-25).pdf | Platon_ Sophistes | Martin Heidegger | 否 |
| 170 | `Authors/Heidegger, Martin/Martin Heidegger - Prolegomena zur Geschichte des Zeitbegriffs (Sommersemester 1925).pdf` | Martin Heidegger - Prolegomena zur Geschichte des Zeitbegriffs (Sommersemester 1925).pdf | Prolegomena zur Geschichte des Zeitbegriffs | Martin Heidegger | 否 |
| 171 | `Authors/Heidegger, Martin/Martin Heidegger - Schellings Abhandlung über das Wesen der menschlichen Freiheit 1809.pdf` | Martin Heidegger - Schellings Abhandlung über das Wesen der menschlichen Freiheit 1809.pdf | Schellings Abhandlung über das Wesen der menschlichen Freiheit 1809 | Martin Heidegger | 否 |
| 172 | `Authors/Heidegger, Martin/Martin Heidegger - Sein und Zeit (1927).pdf` | Martin Heidegger - Sein und Zeit (1927).pdf | Sein und Zeit | Martin Heidegger | 否 |
| 173 | `Authors/Heidegger, Martin/Martin Heidegger - Seminare.pdf` | Martin Heidegger - Seminare.pdf | Seminare | Martin Heidegger | 否 |
| 174 | `Authors/Heidegger, Martin/Martin Heidegger - Seminare_ Kant-Leibniz-Schiller.pdf` | Martin Heidegger - Seminare_ Kant-Leibniz-Schiller.pdf | Seminare_ Kant-Leibniz-Schiller | Martin Heidegger | 否 |
| 175 | `Authors/Heidegger, Martin/Martin Heidegger - Unterwegs zur Sprache (1950-1959).pdf` | Martin Heidegger - Unterwegs zur Sprache (1950-1959).pdf | Unterwegs zur Sprache (1950-1959) | Martin Heidegger | 否 |
| 176 | `Authors/Heidegger, Martin/Martin Heidegger - Wegmarken (1919-1961).pdf` | Martin Heidegger - Wegmarken (1919-1961).pdf | Wegmarken (1919-1961) | Martin Heidegger | 否 |
| 177 | `Authors/Heidegger, Martin/Martin Heidegger - Zu Ernst Jünger.pdf` | Martin Heidegger - Zu Ernst Jünger.pdf | Zu Ernst Jünger | Martin Heidegger | 否 |
| 178 | `Authors/Heidegger, Martin/Martin Heidegger - Zu Hölderlin — Griechenlandreisen.pdf` | Martin Heidegger - Zu Hölderlin — Griechenlandreisen.pdf | Zu Hölderlin — Griechenlandreisen | Martin Heidegger | 否 |
| 179 | `Authors/Heidegger, Martin/Martin Heidegger - Zum Wesen der Sprache und Zur Frage nach der Kunst.pdf` | Martin Heidegger - Zum Wesen der Sprache und Zur Frage nach der Kunst.pdf | Zum Wesen der Sprache und Zur Frage nach der Kunst | Martin Heidegger | 否 |
| 180 | `Authors/Heidegger, Martin/Martin Heidegger - Zur Auslegung von Nietzsches II. Unzeitgemäßer Betrachtung.pdf` | Martin Heidegger - Zur Auslegung von Nietzsches II. Unzeitgemäßer Betrachtung.pdf | Zur Auslegung von Nietzsches II. Unzeitgemäßer Betrachtung | Martin Heidegger | 否 |
| 181 | `Authors/Heidegger, Martin/Martin Heidegger - Über den Anfang (1941).pdf` | Martin Heidegger - Über den Anfang (1941).pdf | Über den Anfang | Martin Heidegger | 否 |
| 182 | `Authors/Heidegger, Martin/Martin Heidegger, Günter Seubold (ed.) - Logik als die Frage nach dem Wesen der Sprache.pdf` | Martin Heidegger, Günter Seubold (ed.) - Logik als die Frage nach dem Wesen der Sprache.pdf | Günter Seubold | Martin Heidegger | 否 |
| 183 | `Authors/Heidegger, Martin/Martin Heidegger, H. Vetter - Geschichte der Philosophie von Thomas von Aquin bis Kant (Winter` | Martin Heidegger, H. Vetter - Geschichte der Philosophie von Thomas von Aquin bis Kant (Wintersemester 1926–27).pdf | Geschichte der Philosophie von Thomas von Aquin bis Kant | Martin Heidegger\|H. Vetter | 否 |
| 184 | `Authors/Heidegger, Martin/Martin Heidegger, Hermann Heidegger - Reden und andere Zeugnisse eines Lebensweges.pdf` | Martin Heidegger, Hermann Heidegger - Reden und andere Zeugnisse eines Lebensweges.pdf | Hermann Heidegger | Martin Heidegger | 否 |
| 185 | `Authors/Heidegger, Martin/Martin Heidegger, Hermann Heidegger - Überlieferte Sprache und technische Sprache-Erker (1989)` | Martin Heidegger, Hermann Heidegger - Überlieferte Sprache und technische Sprache-Erker (1989).pdf | Hermann Heidegger | Martin Heidegger | 否 |
| 186 | `Authors/Heidegger, Martin/Martin Heidegger, Mark Michalski - Seminare. Platon - Aristoteles - Augustinus.pdf` | Martin Heidegger, Mark Michalski - Seminare. Platon - Aristoteles - Augustinus.pdf | Mark Michalski - Seminare. Platon - Aristoteles | Martin Heidegger | 否 |
| 187 | `Authors/Heidegger, Martin/Martin Heidegger, Peter Trawny - Anmerkungen I-V.pdf` | Martin Heidegger, Peter Trawny - Anmerkungen I-V.pdf | Peter Trawny | Martin Heidegger | 否 |
| 188 | `Authors/Heidegger, Martin/Martin Heidegger, Peter Trawny - Seminare_ Hegel-Schelling.pdf` | Martin Heidegger, Peter Trawny - Seminare_ Hegel-Schelling.pdf | Peter Trawny | Martin Heidegger | 否 |
| 189 | `Authors/Heidegger, Martin/Martin Heidegger, Peter Trawny - Überlegungen VII - XI_.pdf` | Martin Heidegger, Peter Trawny - Überlegungen VII - XI_.pdf | Peter Trawny - Überlegungen VII | Martin Heidegger | 否 |
| 190 | `Authors/Heidegger, Martin/Martin Heidegger, Peter Trawny - Überlegungen XII - XV_.pdf` | Martin Heidegger, Peter Trawny - Überlegungen XII - XV_.pdf | Peter Trawny - Überlegungen XII | Martin Heidegger | 否 |
| 191 | `Authors/Heidegger, Martin/Martin Heidegger, Peter von Ruckteschell - Nietzsche_ Seminare 1937 und 1944 1. Nietzsches met` | Martin Heidegger, Peter von Ruckteschell - Nietzsche_ Seminare 1937 und 1944 1. Nietzsches metaphysische Grundstellung.p | Nietzsche_ Seminare 1937 und 1944 1. Nietzsches metaphysische Grundstellung | Martin Heidegger | 否 |
| 192 | `Authors/Heidegger, Martin/Martin Heidegger, Walter Biemel (ed.) - Logik_ Die Frage nach der Wahrheit.pdf` | Martin Heidegger, Walter Biemel (ed.) - Logik_ Die Frage nach der Wahrheit.pdf | Walter Biemel | Martin Heidegger | 否 |
| 193 | `Authors/Heller, Agnes/Aesthetics_and_Modernity__Essays_by_Agnes_Heller_Heller,_Agnes;_Rundell,_John_F.pdf` | Aesthetics_and_Modernity__Essays_by_Agnes_Heller_Heller,_Agnes;_Rundell,_John_F.pdf | Aesthetics_and_Modernity__Essays_by | Agnes Heller | 否 |
| 194 | `Authors/Heller, Agnes/The_three_logics_of_modernity_and_the_double_bind_of_the_modern_imagination_Agnes_Heller..pdf` | The_three_logics_of_modernity_and_the_double_bind_of_the_modern_imagination_Agnes_Heller..pdf | The_three_logics_of_modernity_and_the_double_bind_of_the_modern_imagination | Agnes Heller | 否 |
| 195 | `Authors/Hempel, Carl G/Carl G. Hempel (auth.), Wolfgang Spohn (eds.) - Erkenntnis Orientated_ A Centennial Volume for Ru` | Carl G. Hempel (auth.), Wolfgang Spohn (eds.) - Erkenntnis Orientated_ A Centennial Volume for Rudolf Carnap and Hans Re | Erkenntnis Orientated_ A Centennial Volume for Rudolf Carnap and Hans Reichenbach | Carl G. Hempel | 否 |
| 196 | `Authors/Henrich, Dieter/Aesthetic Judgment and the Moral Image of the World_ Studies in Kant (Studies in Kant and German` | Aesthetic Judgment and the Moral Image of the World_ Studies in Kant (Studies in Kant and German Idealism)_Dieter Henric | Aesthetic Judgment and the Moral Image of the World_ Studies in Kant | Dieter Henrich | 否 |
| 197 | `Authors/Husserl, Edmund/(Husserliana_ Edmund Husserl — Gesammelte Werke, Band 14) Edmund Husserl, Iso Kern - Zur Phänome` | (Husserliana_ Edmund Husserl — Gesammelte Werke, Band 14) Edmund Husserl, Iso Kern - Zur Phänomenologie der Intersubjekt | Zur Phänomenologie der Intersubjektivität_ Texte aus dem Nachlaß. Zweiter Teil. 1921-1928 | Edmund Husserl | 否 |
| 198 | `Authors/Husserl, Edmund/(Husserliana_ Edmund Husserl — Gesammelte Werke, Band 15) Edmund Husserl, Iso Kern - Zur Phänome` | (Husserliana_ Edmund Husserl — Gesammelte Werke, Band 15) Edmund Husserl, Iso Kern - Zur Phänomenologie der Intersubjekt | Zur Phänomenologie der Intersubjektivität_ Texte aus dem Nachlaß. Dritter Teil. 1929–1935 | Edmund Husserl | 否 |
| 199 | `Authors/Ihde, Don/Technology and the Lifeworld_ F - Don Ihde.pdf` | Technology and the Lifeworld_ F - Don Ihde.pdf | Technology and the Lifeworld_ F | Don Ihde | 否 |
| 200 | `Authors/Jaspers, Karl/Karl Jaspers - Truth and Symbol-Twayne Publishers (1959).pdf` | Karl Jaspers - Truth and Symbol-Twayne Publishers (1959).pdf | Truth and Symbol-Twayne Publishers | Karl Jaspers | 否 |
| 201 | `Authors/Jaspers, Karl/Karl Jaspers-The Origin and Goal of History-Greenwood Press Reprint (1977).pdf` | Karl Jaspers-The Origin and Goal of History-Greenwood Press Reprint (1977).pdf | The Origin and Goal of History | Karl Jaspers | 否 |
| 202 | `Authors/Jaspers, Karl/【世纪人文·世纪文库·西学部分】08时代的精神状况.pdf` | 【世纪人文·世纪文库·西学部分】08时代的精神状况.pdf | 【世纪人文·世纪文库·西学部分】08时代的精神状况 | Karl Jaspers | 否 |
| 203 | `Authors/Jenkins, Henry/Henry Jenkins - Fans, Bloggers, and Gamers_ Exploring Participatory Culture-New York University P` | Henry Jenkins - Fans, Bloggers, and Gamers_ Exploring Participatory Culture-New York University Press (2006).pdf | Fans, Bloggers, and Gamers_ Exploring Participatory Culture | Henry Jenkins | 否 |
| 204 | `Authors/Johann Wolfgang von Goethe/歌德文集B/歌德文集.第7卷.戏剧.钱春绮等译.pdf` | 歌德文集.第7卷.戏剧.钱春绮等译.pdf | 歌德文集.第7卷.戏剧 | Johann Wolfgang von Goethe | 否 |
| 205 | `Authors/Juul, Jesper/Juul, Jesper - Half-Real_ Video Games between Real Rules and Fictional Worlds (2011) - libgen.li.pd` | Juul, Jesper - Half-Real_ Video Games between Real Rules and Fictional Worlds (2011) - libgen.li.pdf | Half-Real_ Video Games between Real Rules and Fictional Worlds | Jesper Juul | 否 |
| 206 | `Authors/Kant, Immanuel/康德著作全集 第3卷 纯粹理性批判 （第二版）.pdf` | 康德著作全集 第3卷 纯粹理性批判 （第二版）.pdf | 康德著作全集 第3卷 纯粹理性批判 （第二版） | Immanuel Kant | 否 |
| 207 | `Authors/Kitcher, Philip/(Oxford studies in philosophy of science) Philip Kitcher - Science, Truth, and Democracy-Oxford ` | (Oxford studies in philosophy of science) Philip Kitcher - Science, Truth, and Democracy-Oxford University Press (2001). | Science, Truth, and Democracy | Philip Kitcher | 否 |
| 208 | `Authors/Kitcher, Philip/Philip Kitcher - The Ethical Project-Harvard University Press (2011).pdf` | Philip Kitcher - The Ethical Project-Harvard University Press (2011).pdf | The Ethical Project | Philip Kitcher | 否 |
| 209 | `Authors/Kitcher, Philip/Philip Kitcher - The Nature of Mathematical Knowledge (1985).pdf` | Philip Kitcher - The Nature of Mathematical Knowledge (1985).pdf | The Nature of Mathematical Knowledge | Philip Kitcher | 否 |
| 210 | `Authors/Kittler, Friedrich/Gramophone, Film, Typewriter (1999, Stanford University Press).pdf` | Gramophone, Film, Typewriter (1999, Stanford University Press).pdf | Gramophone, Film, Typewriter | Friedrich Kittler | 否 |
| 211 | `Authors/Kittler, Friedrich/Gramophone, Film, Typewriter-Stanford University Press (1999).pdf` | Gramophone, Film, Typewriter-Stanford University Press (1999).pdf | Gramophone, Film, Typewriter | Friedrich Kittler | 否 |
| 212 | `Authors/Kittler, Friedrich/Optical media Berlin lectures 1999-Polity (2012).pdf` | Optical media Berlin lectures 1999-Polity (2012).pdf | Optical media Berlin lectures 1999 | Friedrich Kittler | 否 |
| 213 | `Authors/Kivy, Peter/Antithetical_arts___on_the_ancient_quarrel_between_literature_and_music_Kivy,_Peter.pdf` | Antithetical_arts___on_the_ancient_quarrel_between_literature_and_music_Kivy,_Peter.pdf | Antithetical_arts___on_the_ancient_quarrel_between_literature_and_music | Peter Kivy | 否 |
| 214 | `Authors/Koestler, Arthur/(Martinus Nijhoff Philosophy Library 6) Arthur Koestler (auth.), Denis Dutton, Michael Krausz (` | (Martinus Nijhoff Philosophy Library 6) Arthur Koestler (auth.), Denis Dutton, Michael Krausz (eds.) - The Concept of Cr | The Concept of Creativity in Science and Art | Arthur Koestler | 否 |
| 215 | `Authors/Kracauer, Siegfried/History-The Last Things Before the Last-Markus Wiener Publishers (2013).pdf` | History-The Last Things Before the Last-Markus Wiener Publishers (2013).pdf | History-The Last Things Before the Last-Markus Wiener Publishers | Siegfried Kracauer | 否 |
| 216 | `Authors/Kracauer, Siegfried/Salaried Masses Duty and Distraction in Weimar Germany-Verso (1998), The.pdf` | Salaried Masses Duty and Distraction in Weimar Germany-Verso (1998), The.pdf | Salaried Masses Duty and Distraction in Weimar Germany | Siegfried Kracauer | 否 |
| 217 | `Authors/Kramer, Sibylle/Thinking with Diagrams The Semiotic Basis of Human Cognition.pdf` | Thinking with Diagrams The Semiotic Basis of Human Cognition.pdf | Thinking with Diagrams The Semiotic Basis of Human Cognition | Sibylle Kramer | 否 |
| 218 | `Authors/Lakatos, Imre/证明与反驳  数学发现的逻辑 - Imre Lakatos.pdf` | 证明与反驳  数学发现的逻辑 - Imre Lakatos.pdf | 证明与反驳 数学发现的逻辑 | Imre Lakatos | 否 |
| 219 | `Authors/Lefebvre, Henri/State, Space, World Selected Essays-Univ Of Minnesota Press (2009).pdf` | State, Space, World Selected Essays-Univ Of Minnesota Press (2009).pdf | State, Space, World Selected Essays | Henri Lefebvre | 否 |
| 220 | `Authors/Lefebvre, Henri/Writings on Cities-Wiley-Blackwell (1996).pdf` | Writings on Cities-Wiley-Blackwell (1996).pdf | Writings on Cities | Henri Lefebvre | 否 |
| 221 | `Authors/Levi-Strauss, Claude/Claude Levi-Strauss.pdf` | Claude Levi-Strauss.pdf | Strauss | Claude Levi-Strauss | 否 |
| 222 | `Authors/Levinas, Emmanuel/时间与他者-le-temps-et-l-autre-伊曼努尔-列维纳斯-著-王嘉军-译-拜德雅--annas-archive--libgenrs-nf-2761855.pdf` | 时间与他者-le-temps-et-l-autre-伊曼努尔-列维纳斯-著-王嘉军-译-拜德雅--annas-archive--libgenrs-nf-2761855.pdf | 时间与他者-le-temps-et-l-autre-伊曼努尔-列维纳斯-著-王嘉军-译-拜德雅--annas-archive--libgenrs-nf-2761855 | Emmanuel Levinas | 否 |
| 223 | `Authors/Levinson, Jerrold/Jerrold Levinson (eds.)-Suffering Art Gladly The Paradox of Negative Emotion in Art-Palgrave M` | Jerrold Levinson (eds.)-Suffering Art Gladly The Paradox of Negative Emotion in Art-Palgrave Macmillan UK (2014).pdf | (eds.)-Suffering Art Gladly The Paradox of Negative Emotion in Art | Jerrold Levinson | 否 |
| 224 | `Authors/Levinson, Jerrold/Jerrold Levinson Music, Art, and Metaphysics Essays in Philosophical Aesthetics.pdf` | Jerrold Levinson Music, Art, and Metaphysics Essays in Philosophical Aesthetics.pdf | Music, Art, and Metaphysics Essays in Philosophical Aesthetics | Jerrold Levinson | 否 |
| 225 | `Authors/Levinson, Jerrold/Jerrold Levinson-Aesthetic Pursuits Essays in Philosophy of Art-Oxford University Press (2017)` | Jerrold Levinson-Aesthetic Pursuits Essays in Philosophy of Art-Oxford University Press (2017).pdf | Aesthetic Pursuits Essays in Philosophy of Art | Jerrold Levinson | 否 |
| 226 | `Authors/Levinson, Paul/Digital McLuhan_ A Guide to the Information Millennium -- Paul Levinson -- New edition, 2001 -- R` | Digital McLuhan_ A Guide to the Information Millennium -- Paul Levinson -- New edition, 2001 -- Routledge -- 97802032781 | Digital McLuhan_ A Guide to the Information Millennium | Paul Levinson | 否 |
| 227 | `Authors/Levinson, Paul/Paul Levinson - Borrowed Tides-Tor Books (2001).pdf` | Paul Levinson - Borrowed Tides-Tor Books (2001).pdf | Borrowed Tides-Tor Books | Paul Levinson | 否 |
| 228 | `Authors/Levinson, Paul/Paul Levinson - Cellphone_ The Story of the World's Most Mobile Medium and How It Has Transformed` | Paul Levinson - Cellphone_ The Story of the World's Most Mobile Medium and How It Has Transformed Everything! (2004).pdf | Cellphone_ The Story of the World's Most Mobile Medium and How It Has Transformed Everything! | Paul Levinson | 否 |
| 229 | `Authors/Levinson, Paul/Paul Levinson - The soft edge _ a natural history and future of the information revolution-Routle` | Paul Levinson - The soft edge _ a natural history and future of the information revolution-Routledge (1997).pdf | The soft edge _ a natural history and future of the information revolution | Paul Levinson | 否 |
| 230 | `Authors/Lewis, David/David Lewis On the Plurality of Worlds.pdf` | David Lewis On the Plurality of Worlds.pdf | On the Plurality of Worlds | David Lewis | 否 |
| 231 | `Authors/Ludwig Wittgenstein/Tractatus Logico-Philosophicus _ Side-by-side-by-side edition.pdf` | Tractatus Logico-Philosophicus _ Side-by-side-by-side edition.pdf | Tractatus Logico-Philosophicus _ Side-by-side-by-side edition | Ludwig Wittgenstein | 是 |
| 232 | `Authors/Luhman, Niklas/differentiation of society, The.pdf` | differentiation of society, The.pdf | differentiation of society, The | Niklas Luhman | 否 |
| 233 | `Authors/Luhman, Niklas/社会中的艺术.pdf` | 社会中的艺术.pdf | 社会中的艺术 | Niklas Luhman | 否 |
| 234 | `Authors/Manovich, Lev/Language of New Media, The - Lev Manovich.pdf` | Language of New Media, The - Lev Manovich.pdf | Language of New Media, The | Lev Manovich | 否 |
| 235 | `Authors/Marcuse, Herbert/(现代西方学术文库)  - 审美之维_ 马尔库塞美学论著集-生活·读书·新知三联书店 (1989).pdf` | (现代西方学术文库)  - 审美之维_ 马尔库塞美学论著集-生活·读书·新知三联书店 (1989).pdf | 审美之维_ 马尔库塞美学论著集-生活·读书·新知三联书店 | Herbert Marcuse | 否 |
| 236 | `Authors/Marcuse, Herbert/[二十世纪西方哲学译丛]单向度的人-发达工业社会意识形态研究·[美]赫伯特·马尔库塞 著.pdf` | [二十世纪西方哲学译丛]单向度的人-发达工业社会意识形态研究·[美]赫伯特·马尔库塞 著.pdf | 单向度的人-发达工业社会意识形态研究· | Herbert Marcuse | 否 |
| 237 | `Authors/Marcuse, Herbert/苏联的马克思主义 一种批判的分析-中国人民大学出版社 (2012).pdf` | 苏联的马克思主义 一种批判的分析-中国人民大学出版社 (2012).pdf | 苏联的马克思主义 一种批判的分析-中国人民大学出版社 | Herbert Marcuse | 否 |
| 238 | `Authors/Margolis, Joseph/Joseph Margolis (auth.) - Culture and Cultural Entities_ Toward a New Unity of Science.pdf` | Joseph Margolis (auth.) - Culture and Cultural Entities_ Toward a New Unity of Science.pdf | (auth.) | Joseph Margolis | 否 |
| 239 | `Authors/Margolis, Joseph/Joseph Margolis (auth.), Edmund F. Byrne, Joseph C. Pitt (eds.) - Technological Transformation_` | Joseph Margolis (auth.), Edmund F. Byrne, Joseph C. Pitt (eds.) - Technological Transformation_ Contextual and Conceptua | Technological Transformation_ Contextual and Conceptual Implications | Joseph Margolis\|Edmund F. Byrne\|Joseph C. Pitt | 否 |
| 240 | `Authors/Margolis, Joseph/Joseph Margolis - Pragmatism's Advantage_ American and European Philosophy at the End of the Tw` | Joseph Margolis - Pragmatism's Advantage_ American and European Philosophy at the End of the Twentieth Century  -Stanfor | Pragmatism's Advantage_ American and European Philosophy at the End of the Twentieth Century | Joseph Margolis | 否 |
| 241 | `Authors/Margolis, Joseph/Joseph Margolis - Selves and Other Texts_ The Case for Cultural Realism-Pennsylvania State Univ` | Joseph Margolis - Selves and Other Texts_ The Case for Cultural Realism-Pennsylvania State Univ Pr (Txt) (2001).pdf | Selves and Other Texts_ The Case for Cultural Realism-Pennsylvania State Univ Pr | Joseph Margolis | 否 |
| 242 | `Authors/Marion, Jean-Luc/The_Visible_and_the_Revealed_(Perspectives_in_Continental_Philosophy)_Jean-Luc_Marion.pdf` | The_Visible_and_the_Revealed_(Perspectives_in_Continental_Philosophy)_Jean-Luc_Marion.pdf | The_Visible_and_the_Revealed | Jean-Luc Marion | 否 |
| 243 | `Authors/Martin, Reinhold/Pedro Fiori Arantes_ Adriana Kauffmann_ Timothy Frye_ Reinhold Martin - The Rent of Form_ Archi` | Pedro Fiori Arantes_ Adriana Kauffmann_ Timothy Frye_ Reinhold Martin - The Rent of Form_ Architecture and Labor in the  | The Rent of Form_ Architecture and Labor in the Digital Age | Reinhold Martin | 否 |
| 244 | `Authors/Martin, Reinhold/The Organizational Complex_ Architecture, Media, and -- Reinhold Martin -- 2003 -- The MIT Pres` | The Organizational Complex_ Architecture, Media, and -- Reinhold Martin -- 2003 -- The MIT Press -- 9780262134262 -- 449 | The Organizational Complex_ Architecture, Media, and | Reinhold Martin | 否 |
| 245 | `Authors/Martin, Reinhold/Utopia’s Ghost_ Architecture and Postmodernism, Again -- Reinhold Martin -- 2010 -- University ` | Utopia’s Ghost_ Architecture and Postmodernism, Again -- Reinhold Martin -- 2010 -- University of Minnesota Press -- 978 | Utopia’s Ghost_ Architecture and Postmodernism, Again | Reinhold Martin | 否 |
| 246 | `Authors/McLuhan, Marshall/Marshall McLuhan, W. Terrence Gordon - Counterblast  -Gingko Press, Inc (2011).pdf` | Marshall McLuhan, W. Terrence Gordon - Counterblast  -Gingko Press, Inc (2011).pdf | Counterblast | Marshall McLuhan\|W. Terrence Gordon | 否 |
| 247 | `Authors/McLuhan, Marshall/Marshall McLuhan_ Wilfred Watson - From cliche to archetype-Viking Press (1970).pdf` | Marshall McLuhan_ Wilfred Watson - From cliche to archetype-Viking Press (1970).pdf | Wilfred Watson | Marshall McLuhan | 否 |
| 248 | `Authors/McLuhan, Marshall/[加拿大] 马歇尔·麦克卢汉 - 机器新娘_工业人的民俗-中国人民大学出版社 (2004).pdf` | [加拿大] 马歇尔·麦克卢汉 - 机器新娘_工业人的民俗-中国人民大学出版社 (2004).pdf | 机器新娘_工业人的民俗-中国人民大学出版社 | Marshall McLuhan | 否 |
| 249 | `Authors/Menke, Christoph/Menke, Christoph - Die Kraft der Kunst-Suhrkamp (2013).epub` | Menke, Christoph - Die Kraft der Kunst-Suhrkamp (2013).epub | Die Kraft der Kunst-Suhrkamp (2013).epub | Christoph Menke | 否 |
| 250 | `Authors/Merleau-Ponty, Moris/眼与心.pdf` | 眼与心.pdf | 眼与心 | Moris Merleau-Ponty | 否 |
| 251 | `Authors/Nagel, Ernest/Nagel, Ernest - The Structure of Science_ Problems in the Logic of Scientific Explanation-Harcourt` | Nagel, Ernest - The Structure of Science_ Problems in the Logic of Scientific Explanation-Harcourt, Brace & World, INC.  | The Structure of Science_ Problems in the Logic of Scientific Explanation-Harcourt, Brace & World, INC | Ernest Nagel | 否 |
| 252 | `Authors/Nagel, Thomas/Thomas Nagel - The view from nowhere.pdf` | Thomas Nagel - The view from nowhere.pdf | The view from nowhere | Thomas Nagel | 否 |
| 253 | `Authors/Negri, Antonio/(Heretical Thought) Michael Hardt, Antonio Negri - Assembly-Oxford University Press (2017).pdf` | (Heretical Thought) Michael Hardt, Antonio Negri - Assembly-Oxford University Press (2017).pdf | Assembly | Michael Hardt\|Antonio Negri | 否 |
| 254 | `Authors/Noe, Alva/The Entanglement_ How Art and Philosophy Make Us What We Are_Alva Noë.pdf` | The Entanglement_ How Art and Philosophy Make Us What We Are_Alva Noë.pdf | The Entanglement_ How Art and Philosophy Make Us What We Are | Alva Noe | 否 |
| 255 | `Authors/Novalis/Novalis Schriften.pdf` | Novalis Schriften.pdf | Schriften | Novalis | 否 |
| 256 | `Authors/Nussbaum, Martha/Martha C. Nussbaum, Amelie Oksenberg Rorty - Essays on Aristotle's De Anima.pdf` | Martha C. Nussbaum, Amelie Oksenberg Rorty - Essays on Aristotle's De Anima.pdf | C. Nussbaum, Amelie Oksenberg Rorty | Martha Nussbaum | 否 |
| 257 | `Authors/Nussbaum, Martha/Martha C. Nussbaum, Saul Levmore - Aging Thoughtfully_ Conversations about Retirement, Romance,` | Martha C. Nussbaum, Saul Levmore - Aging Thoughtfully_ Conversations about Retirement, Romance, Wrinkles, and Regret.pdf | Aging Thoughtfully_ Conversations about Retirement, Romance, Wrinkles, and Regret | Martha C. Nussbaum\|Saul Levmore | 否 |
| 258 | `Authors/Nussbaum, Martha/Nussbaum, Martha C - Women and Human Development _ The Capabilities Approach.pdf` | Nussbaum, Martha C - Women and Human Development _ The Capabilities Approach.pdf | Nussbaum, Martha C | Martha Nussbaum | 否 |
| 259 | `Authors/Peters, John Durham/Kenneth Cmiel_ John Durham Peters - Promiscuous Knowledge_ Information, Image, and Other Tru` | Kenneth Cmiel_ John Durham Peters - Promiscuous Knowledge_ Information, Image, and Other Truth Games in History (2020, U | Promiscuous Knowledge_ Information, Image, and Other Truth Games in History | John Durham Peters | 否 |
| 260 | `Authors/Pinker, Steven/The Blank Slate. Modern Denial of Human Nature [html,ugly]_Steven Pinker.pdf` | The Blank Slate. Modern Denial of Human Nature [html,ugly]_Steven Pinker.pdf | The Blank Slate. Modern Denial of Human Nature [html,ugly] | Steven Pinker | 否 |
| 261 | `Authors/Pinker, Steven/Vision_ A Computational Investigation into the Human Representation and Processing of Visual Info` | Vision_ A Computational Investigation into the Human Representation and Processing of Visual Information  _David Marr.pd | Vision_ A Computational Investigation into the Human Representation and Processing of Visual Information _David Marr | Steven Pinker | 否 |
| 262 | `Authors/Poulantzas, Nicos/Nicos Poulantzas - State, Power, Socialism (2001).pdf` | Nicos Poulantzas - State, Power, Socialism (2001).pdf | State, Power, Socialism | Nicos Poulantzas | 否 |
| 263 | `Authors/Quine, Willard Van Orman/Quine, Quine Willard Van Orman - Philosophy of logic-Harvard University Press (1986).pd` | Quine, Quine Willard Van Orman - Philosophy of logic-Harvard University Press (1986).pdf | Philosophy of logic | Willard Van Orman Quine | 否 |
| 264 | `Authors/Ranciere, Jacques/(当代激进思想家译丛) [法] 雅克·朗西埃 - 文学的政治-南京大学出版社 (2014).pdf` | (当代激进思想家译丛) [法] 雅克·朗西埃 - 文学的政治-南京大学出版社 (2014).pdf | 文学的政治-南京大学出版社 | Jacques Rancière | 否 |
| 265 | `Authors/Ranciere, Jacques/(当代激进思想家译丛) 雅克·朗西埃 - 审美无意识-南京大学出版社 (2020).pdf` | (当代激进思想家译丛) 雅克·朗西埃 - 审美无意识-南京大学出版社 (2020).pdf | 审美无意识-南京大学出版社 | Jacques Rancière | 否 |
| 266 | `Authors/Ranciere, Jacques/Aisthesis Scenes from the Aesthetic Regime of Art-Verso (2013).pdf` | Aisthesis Scenes from the Aesthetic Regime of Art-Verso (2013).pdf | Aisthesis Scenes from the Aesthetic Regime of Art | Jacques Ranciere | 否 |
| 267 | `Authors/Ranciere, Jacques/Jacques Ranciere - Aesthetics and Its Discontents (2009.pdf` | Jacques Ranciere - Aesthetics and Its Discontents (2009.pdf | Aesthetics and Its Discontents (2009 | Jacques Rancière | 否 |
| 268 | `Authors/Ranciere, Jacques/Jacques Ranciere - Hatred of Democracy (2009).pdf` | Jacques Ranciere - Hatred of Democracy (2009).pdf | Hatred of Democracy | Jacques Rancière | 否 |
| 269 | `Authors/Ranciere, Jacques/Jacques Ranciere - The Emancipated Spectator (2009).pdf` | Jacques Ranciere - The Emancipated Spectator (2009).pdf | The Emancipated Spectator | Jacques Rancière | 否 |
| 270 | `Authors/Ranciere, Jacques/Jacques Ranciere - The Flesh of Words The Politics of Writing (2004).pdf` | Jacques Ranciere - The Flesh of Words The Politics of Writing (2004).pdf | The Flesh of Words The Politics of Writing | Jacques Rancière | 否 |
| 271 | `Authors/Ranciere, Jacques/Jacques Ranciere - The Future of the Image (2007).pdf` | Jacques Ranciere - The Future of the Image (2007).pdf | The Future of the Image | Jacques Rancière | 否 |
| 272 | `Authors/Ranciere, Jacques/Jacques Ranciere - The Nights Of Labor The Workers Dream in Nineteenth-Century France (1991).p` | Jacques Ranciere - The Nights Of Labor The Workers Dream in Nineteenth-Century France (1991).pdf | The Nights Of Labor The Workers Dream in Nineteenth-Century France | Jacques Rancière | 否 |
| 273 | `Authors/Ricoeur, Paul/(Marquette Studies in Philosophy 10) Paul Ricoeur-A Key to Husserl's Ideas I-Marquette University ` | (Marquette Studies in Philosophy 10) Paul Ricoeur-A Key to Husserl's Ideas I-Marquette University Press (1996).pdf | A Key to Husserl's Ideas I | Paul Ricoeur | 否 |
| 274 | `Authors/Ricoeur, Paul/Paul Ricoeur-Time and Narrative, Volume 3 (Time & Narrative)-University Of Chicago Press (1988).pd` | Paul Ricoeur-Time and Narrative, Volume 3 (Time & Narrative)-University Of Chicago Press (1988).pdf | Time and Narrative, Volume 3 | Paul Ricoeur | 否 |
| 275 | `Authors/Ricoeur, Paul/作为一个他者的自身-商务印书馆 (2013).pdf` | 作为一个他者的自身-商务印书馆 (2013).pdf | 作为一个他者的自身-商务印书馆 | Paul Ricoeur | 否 |
| 276 | `Authors/Rorty, Richard/Philosophy as Cultural Politics - Richard Rorty.pdf` | Philosophy as Cultural Politics - Richard Rorty.pdf | Philosophy as Cultural Politics | Richard Rorty | 否 |
| 277 | `Authors/Rorty, Richard/The Linguistic Turn_ Essays in - Richard Rorty.pdf` | The Linguistic Turn_ Essays in - Richard Rorty.pdf | The Linguistic Turn_ Essays in | Richard Rorty | 否 |
| 278 | `Authors/Roy, Arundhati/Arundhati Roy - Azadi_ Freedom. Fascism. Fiction.-Haymarket Books (2020).pdf` | Arundhati Roy - Azadi_ Freedom. Fascism. Fiction.-Haymarket Books (2020).pdf | Azadi_ Freedom. Fascism. Fiction.-Haymarket Books | Arundhati Roy | 否 |
| 279 | `Authors/Roy, Arundhati/Arundhati Roy - Power Politics-South End Press (2001).pdf` | Arundhati Roy - Power Politics-South End Press (2001).pdf | Power Politics | Arundhati Roy | 否 |
| 280 | `Authors/Roy, Arundhati/Arundhati Roy - The God of Small Things-Harper Perennial (1998).pdf` | Arundhati Roy - The God of Small Things-Harper Perennial (1998).pdf | The God of Small Things-Harper Perennial | Arundhati Roy | 否 |
| 281 | `Authors/Roy, Arundhati/Arundhati Roy - The Ministry of Utmost Happiness-Penguin Books (2017).pdf` | Arundhati Roy - The Ministry of Utmost Happiness-Penguin Books (2017).pdf | The Ministry of Utmost Happiness | Arundhati Roy | 否 |
| 282 | `Authors/Roy, Arundhati/Arundhati Roy - Walking with the comrades-Penguin Books (2014).pdf` | Arundhati Roy - Walking with the comrades-Penguin Books (2014).pdf | Walking with the comrades | Arundhati Roy | 否 |
| 283 | `Authors/Roy, Arundhati/Arundhati Roy, John Cusack - Things That Can and Cannot Be Said_ Essays and Conversations-Haymark` | Arundhati Roy, John Cusack - Things That Can and Cannot Be Said_ Essays and Conversations-Haymarket Books (2016).pdf | Things That Can and Cannot Be Said_ Essays and Conversations-Haymarket Books | Arundhati Roy\|John Cusack | 否 |
| 284 | `Authors/Ryan, Marie-Laura/Marie-Laure Ryan - Possible Worlds, Artificial lntelligence, and Narrative Theory (1991, India` | Marie-Laure Ryan - Possible Worlds, Artificial lntelligence, and Narrative Theory (1991, Indiana University Press) - lib | Possible Worlds, Artificial lntelligence, and Narrative Theory | Marie-Laura Ryan | 否 |
| 285 | `Authors/Said, Edward W/Edward W Said - Orientalism.pdf` | Edward W Said - Orientalism.pdf | Orientalism | Edward W. Said | 否 |
| 286 | `Authors/Schapiro, Meyer/Words_and_Pictures._On_the_Literal_and_the_Symbolic_in_the_Illustration_of_a_Text_Meyer_Schapiro` | Words_and_Pictures._On_the_Literal_and_the_Symbolic_in_the_Illustration_of_a_Text_Meyer_Schapiro.pdf | Words_and_Pictures._On_the_Literal_and_the_Symbolic_in_the_Illustration_of_a_Text | Meyer Schapiro | 否 |
| 287 | `Authors/Schelling, FWJ/谢林著作集 论人类自由的本质及相关对象.pdf` | 谢林著作集 论人类自由的本质及相关对象.pdf | 谢林著作集 论人类自由的本质及相关对象 | FWJ Schelling | 否 |
| 288 | `Authors/Schiller, Dan/信息拜物教-批判与解构-丹-席勒-2008--annas-archive--libgenrs-nf-2582978.pdf` | 信息拜物教-批判与解构-丹-席勒-2008--annas-archive--libgenrs-nf-2582978.pdf | 信息拜物教-批判与解构-丹-席勒-2008--annas-archive--libgenrs-nf-2582978 | Dan Schiller | 否 |
| 289 | `Authors/Schlegel, Friedrich von/August Wilhelm Schlegel (Hrsg.), Friedrich Schlegel (Hrsg.) - Athenaeum. 1798-1800-Cotta` | August Wilhelm Schlegel (Hrsg.), Friedrich Schlegel (Hrsg.) - Athenaeum. 1798-1800-Cotta (1960).pdf | Athenaeum. 1798-1800-Cotta | August Wilhelm Schlegel\|Friedrich Schlegel | 否 |
| 290 | `Authors/Shusterman, Richard/Ars Erotica Sex and Somaesthetics in the Classical Arts of Love-Cambridge University Press (` | Ars Erotica Sex and Somaesthetics in the Classical Arts of Love-Cambridge University Press (2021).pdf | Ars Erotica Sex and Somaesthetics in the Classical Arts of Love | Richard Shusterman | 否 |
| 291 | `Authors/Shusterman, Richard/Body Consciousness A Philosophy of Mindfulness and Somaesthetics-Cambridge University Press ` | Body Consciousness A Philosophy of Mindfulness and Somaesthetics-Cambridge University Press (2008).pdf | Body Consciousness A Philosophy of Mindfulness and Somaesthetics | Richard Shusterman | 否 |
| 292 | `Authors/Siegert, Bernhard/Lorenz Engell_ Bernhard Siegert - Entwerfen_ Zeitschrift für Medien- und Kulturforschung (2012` | Lorenz Engell_ Bernhard Siegert - Entwerfen_ Zeitschrift für Medien- und Kulturforschung (2012, Meiner, F) - libgen.li.p | Entwerfen_ Zeitschrift für Medien- und Kulturforschung (2012, Meiner, F) | Bernhard Siegert | 否 |
| 293 | `Authors/Siegert, Bernhard/Lorenz Engell_ Bernhard Siegert - Producing Places_ Zeitschrift für Medien- und Kulturforschun` | Lorenz Engell_ Bernhard Siegert - Producing Places_ Zeitschrift für Medien- und Kulturforschung, Heft 5_1_2014-Meiner, F | Producing Places_ Zeitschrift für Medien- und Kulturforschung, Heft 5_1_2014-Meiner, F | Lorenz Engell_ Bernhard Siegert | 否 |
| 294 | `Authors/Simanowski, Roberto/Sociophobia Political Change in the Digital Utopia ( etc.) (Z-Library).pdf` | Sociophobia Political Change in the Digital Utopia ( etc.) (Z-Library).pdf | Sociophobia Political Change in the Digital Utopia ( etc.) | Roberto Simanowski | 否 |
| 295 | `Authors/Simondon, Gilbert/Gilbert Simondon-Du mode d'existence des objets techniques-Aubier (2001).pdf` | Gilbert Simondon-Du mode d'existence des objets techniques-Aubier (2001).pdf | Du mode d'existence des objets techniques-Aubier | Gilbert Simondon | 否 |
| 296 | `Authors/Sloterdijk, Peter/Peter Sloterdijk - You Must Change Your Life_ On Anthropotechnics-Polity (2013).pdf` | Peter Sloterdijk - You Must Change Your Life_ On Anthropotechnics-Polity (2013).pdf | You Must Change Your Life_ On Anthropotechnics | Peter Sloterdijk | 否 |
| 297 | `Authors/Sperber, Dan/Meaning and relevance_Sperber, Dan;Wilson, Deirdre.pdf` | Meaning and relevance_Sperber, Dan;Wilson, Deirdre.pdf | Meaning and relevance | Dan Sperber | 否 |
| 298 | `Authors/Stiegler, Bernard/技术与时间3 斯蒂格勒.pdf` | 技术与时间3 斯蒂格勒.pdf | 技术与时间3 斯蒂格勒 | Bernard Stiegler | 否 |
| 299 | `Authors/Strauss, Leo/(Phoenix Books) Leo Strauss, Elsa M. Sinclair - The Political Philosophy of Hobbes_ Its Basis and I` | (Phoenix Books) Leo Strauss, Elsa M. Sinclair - The Political Philosophy of Hobbes_ Its Basis and Its Genesis -Universit | The Political Philosophy of Hobbes_ Its Basis and Its Genesis | Leo Strauss | 否 |
| 300 | `Authors/Strauss, Leo/Leo Strauss - An Introduction to Political Philosophy_ Ten Essays-Wayne State University Press (198` | Leo Strauss - An Introduction to Political Philosophy_ Ten Essays-Wayne State University Press (1989).pdf | An Introduction to Political Philosophy_ Ten Essays | Leo Strauss | 否 |
| 301 | `Authors/Strauss, Leo/Leo Strauss - The Argument and the Action of Plato's Laws-University Of Chicago Press (1975).pdf` | Leo Strauss - The Argument and the Action of Plato's Laws-University Of Chicago Press (1975).pdf | The Argument and the Action of Plato's Laws | Leo Strauss | 否 |
| 302 | `Authors/Turkle, Sherry/Alone_Together__Why_We_Expect_More_From_Technology_and_Less_From_Each_Other_Sherry_Turkle.pdf` | Alone_Together__Why_We_Expect_More_From_Technology_and_Less_From_Each_Other_Sherry_Turkle.pdf | Alone_Together__Why_We_Expect_More_From_Technology_and_Less_From_Each_Other | Sherry Turkle | 否 |
| 303 | `Authors/Virilio, Paul/Accident of Art (2005), The.pdf` | Accident of Art (2005), The.pdf | Accident of Art (2005), The | Paul Virilio | 否 |
| 304 | `Authors/Virilio, Paul/Aesthetics of Disappearance-Semiotext(e) (1991), The.pdf` | Aesthetics of Disappearance-Semiotext(e) (1991), The.pdf | Aesthetics of Disappearance-Semiotext(e) (1991), The | Paul Virilio | 否 |
| 305 | `Authors/Virilio, Paul/Bunker Archaeology-Princeton Architectural Press (2008).pdf` | Bunker Archaeology-Princeton Architectural Press (2008).pdf | Bunker Archaeology | Paul Virilio | 否 |
| 306 | `Authors/Virilio, Paul/El Cibermundo, La Politica De Lo Peor (Teorema Serie Menor) (Spanish Edition) (2004).pdf` | El Cibermundo, La Politica De Lo Peor (Teorema Serie Menor) (Spanish Edition) (2004).pdf | El Cibermundo, La Politica De Lo Peor | Paul Virilio | 否 |
| 307 | `Authors/Virilio, Paul/Vision Machine, The.pdf` | Vision Machine, The.pdf | Vision Machine, The | Paul Virilio | 否 |
| 308 | `Authors/Watson, Gary/Gary Watson - Agency and Answerability_ Selected Essays (2004).pdf` | Gary Watson - Agency and Answerability_ Selected Essays (2004).pdf | Agency and Answerability_ Selected Essays | Gary Watson | 否 |
| 309 | `Authors/Welsch, Wolfgang/Wolfgang Welsch - Unsere postmoderne Moderne-Akademie Verlag (2008).pdf` | Wolfgang Welsch - Unsere postmoderne Moderne-Akademie Verlag (2008).pdf | Unsere postmoderne Moderne-Akademie Verlag | Wolfgang Welsch | 否 |
| 310 | `Authors/Welsch, Wolfgang/我们的后现代的现代 (（德）沃尔夫冈·韦尔施（Wolfgang Welsch）著；洪天富译 etc.) (Z-Library).pdf` | 我们的后现代的现代 (（德）沃尔夫冈·韦尔施（Wolfgang Welsch）著；洪天富译 etc.) (Z-Library).pdf | 我们的后现代的现代 | Wolfgang Welsch | 否 |
| 311 | `Authors/Williams, Raymond/Raymond Williams - Culture and Society, 1780-1950  -Doubleday Anchor (1958).pdf` | Raymond Williams - Culture and Society, 1780-1950  -Doubleday Anchor (1958).pdf | Culture and Society, 1780-1950 -Doubleday Anchor | Raymond Williams | 否 |
| 312 | `Authors/Zizek, Slavoj/[斯洛文尼亚] 斯拉沃热·齐泽克（Slavoj Žižek） - 易碎的绝对.pdf` | [斯洛文尼亚] 斯拉沃热·齐泽克（Slavoj Žižek） - 易碎的绝对.pdf | 易碎的绝对 | Slavoj Žižek | 否 |
| 313 | `Authors/Zizek, Slavoj/[斯洛文尼亚] 斯拉沃热·齐泽克（Slavoj Žižek） - 有人说过集权主义吗？.pdf` | [斯洛文尼亚] 斯拉沃热·齐泽克（Slavoj Žižek） - 有人说过集权主义吗？.pdf | 有人说过集权主义吗？ | Slavoj Zizek | 否 |
| 314 | `Authors/Zizek, Slavoj/[斯诺文尼亚] 斯拉沃热·齐泽克 - 突破可能性的极限-福建教育出版社有限责任公司 (2017).pdf` | [斯诺文尼亚] 斯拉沃热·齐泽克 - 突破可能性的极限-福建教育出版社有限责任公司 (2017).pdf | 突破可能性的极限-福建教育出版社有限责任公司 | Slavoj Žižek | 否 |
| 315 | `Authors/Zizek, Slavoj/齐泽克 - 享受你的症状：好莱坞内外的拉康.pdf` | 齐泽克 - 享受你的症状：好莱坞内外的拉康.pdf | 享受你的症状：好莱坞内外的拉康 | Slavoj Žižek | 否 |
| 316 | `Authors/de Certeau, Michel/The Practice of Everyday Life, Vol. 2_ Living and Cooking  _Michel de Certeau, Luce Giard, Pi` | The Practice of Everyday Life, Vol. 2_ Living and Cooking  _Michel de Certeau, Luce Giard, Pierre Mayol.pdf | The Practice of Everyday Life, Vol. 2_ Living and Cooking | Michel de Certeau | 否 |
| 317 | `Authors/von Foerster, Heinz/Heinz Von Foerster - Understanding Understanding_ Essays on Cybernetics and Cognition.pdf` | Heinz Von Foerster - Understanding Understanding_ Essays on Cybernetics and Cognition.pdf | Understanding Understanding_ Essays on Cybernetics and Cognition | Heinz von Foerster | 否 |
| 318 | `Authors/von Uexkull, Jakob/Jakob von Uexküll - A Foray Into the Worlds of Animals and Humans_ With a Theory of Meaning-U` | Jakob von Uexküll - A Foray Into the Worlds of Animals and Humans_ With a Theory of Meaning-U of Minnesota Press (2010). | A Foray Into the Worlds of Animals and Humans_ With a Theory of Meaning | Jakob von Uexküll | 否 |
| 319 | `Authors/余英时/余英时：中国知识人之史的考察.pdf` | 余英时：中国知识人之史的考察.pdf | 余英时：中国知识人之史的考察 | 余英时 | 否 |
| 320 | `Authors/余英时/犹记风吹水上鳞：钱穆与现代中国学术.余英时.三民书局股份有限公司1991.pdf` | 犹记风吹水上鳞：钱穆与现代中国学术.余英时.三民书局股份有限公司1991.pdf | 犹记风吹水上鳞：钱穆与现代中国学术.余英时.三民书局股份有限公司1991 | 余英时 | 否 |
| 321 | `Authors/唐君毅/唐君毅：《中国哲学原论》原教篇.pdf` | 唐君毅：《中国哲学原论》原教篇.pdf | 唐君毅：《中国哲学原论》原教篇 | 唐君毅 | 否 |
| 322 | `Authors/唐君毅/唐君毅：中国哲学原论·原性篇.pdf` | 唐君毅：中国哲学原论·原性篇.pdf | 唐君毅：中国哲学原论·原性篇 | 唐君毅 | 否 |
| 323 | `Authors/唐君毅/唐君毅：中国哲学原论·原教篇.pdf` | 唐君毅：中国哲学原论·原教篇.pdf | 唐君毅：中国哲学原论·原教篇 | 唐君毅 | 否 |
| 324 | `Authors/唐君毅/唐君毅：中国哲学原论·导论篇.pdf` | 唐君毅：中国哲学原论·导论篇.pdf | 唐君毅：中国哲学原论·导论篇 | 唐君毅 | 否 |
| 325 | `Authors/唐君毅/唐君毅：中国文化之精神价值.pdf` | 唐君毅：中国文化之精神价值.pdf | 唐君毅：中国文化之精神价值 | 唐君毅 | 否 |
| 326 | `Authors/川野洋/ネットワーク美学の誕生——下 - 川野洋.pdf` | ネットワーク美学の誕生——下 - 川野洋.pdf | ネットワーク美学の誕生——下 | 川野洋 | 否 |
| 327 | `Authors/徐复观/徐復觀-最後雜文集.pdf` | 徐復觀-最後雜文集.pdf | 徐復觀-最後雜文集 | 徐复观 | 否 |
| 328 | `Authors/徐复观/徐復觀-雜文續集.pdf` | 徐復觀-雜文續集.pdf | 徐復觀-雜文續集 | 徐复观 | 否 |
| 329 | `Authors/徐复观/徐復觀·中國人性論史.pdf` | 徐復觀·中國人性論史.pdf | 徐復觀·中國人性論史 | 徐复观 | 否 |
| 330 | `Authors/徐复观/徐復觀·中國思想史論集續篇.pdf` | 徐復觀·中國思想史論集續篇.pdf | 徐復觀·中國思想史論集續篇 | 徐复观 | 否 |
| 331 | `Authors/徐复观/徐復觀·中國文學論集.pdf` | 徐復觀·中國文學論集.pdf | 徐復觀·中國文學論集 | 徐复观 | 否 |
| 332 | `Authors/徐复观/徐復觀·中國文學論集續編.pdf` | 徐復觀·中國文學論集續編.pdf | 徐復觀·中國文學論集續編 | 徐复观 | 否 |
| 333 | `Authors/徐复观/徐復觀·中國經學史的基礎.pdf` | 徐復觀·中國經學史的基礎.pdf | 徐復觀·中國經學史的基礎 | 徐复观 | 否 |
| 334 | `Authors/徐复观/徐復觀·中國藝術精神.pdf` | 徐復觀·中國藝術精神.pdf | 徐復觀·中國藝術精神 | 徐复观 | 否 |
| 335 | `Authors/徐复观/徐復觀·儒家政治思想與民主自由人權.pdf` | 徐復觀·儒家政治思想與民主自由人權.pdf | 徐復觀·儒家政治思想與民主自由人權 | 徐复观 | 否 |
| 336 | `Authors/徐复观/徐復觀·兩漢思想史 卷一.pdf` | 徐復觀·兩漢思想史 卷一.pdf | 徐復觀·兩漢思想史 卷一 | 徐复观 | 否 |
| 337 | `Authors/徐复观/徐復觀·兩漢思想史 卷三.pdf` | 徐復觀·兩漢思想史 卷三.pdf | 徐復觀·兩漢思想史 卷三 | 徐复观 | 否 |
| 338 | `Authors/徐复观/徐復觀·公孫龍子講疏.pdf` | 徐復觀·公孫龍子講疏.pdf | 徐復觀·公孫龍子講疏 | 徐复观 | 否 |
| 339 | `Authors/徐复观/徐復觀·周官成立之時代及其思想性格.pdf` | 徐復觀·周官成立之時代及其思想性格.pdf | 徐復觀·周官成立之時代及其思想性格 | 徐复观 | 否 |
| 340 | `Authors/徐复观/徐復觀·憶往事.pdf` | 徐復觀·憶往事.pdf | 徐復觀·憶往事 | 徐复观 | 否 |
| 341 | `Authors/徐复观/徐復觀·新版学术与政治之间 甲乙集.pdf` | 徐復觀·新版学术与政治之间 甲乙集.pdf | 徐復觀·新版学术与政治之间 甲乙集 | 徐复观 | 否 |
| 342 | `Authors/徐复观/徐復觀·看世局.pdf` | 徐復觀·看世局.pdf | 徐復觀·看世局 | 徐复观 | 否 |
| 343 | `Authors/徐复观/徐復觀·記所思.pdf` | 徐復觀·記所思.pdf | 徐復觀·記所思 | 徐复观 | 否 |
| 344 | `Authors/徐复观/徐復觀·論中共.pdf` | 徐復觀·論中共.pdf | 徐復觀·論中共 | 徐复观 | 否 |
| 345 | `Authors/徐复观/徐復觀·黃大癡兩山水長卷的真偽問題.pdf` | 徐復觀·黃大癡兩山水長卷的真偽問題.pdf | 徐復觀·黃大癡兩山水長卷的真偽問題 | 徐复观 | 否 |
| 346 | `Authors/李泽厚/中国古代思想史论.李泽厚著.三联书店2008.pdf` | 中国古代思想史论.李泽厚著.三联书店2008.pdf | 中国古代思想史论 | 李泽厚 | 否 |
| 347 | `Authors/李泽厚/中国现代思想史论.李泽厚著.三联书店2008.pdf` | 中国现代思想史论.李泽厚著.三联书店2008.pdf | 中国现代思想史论 | 李泽厚 | 否 |
| 348 | `Authors/李泽厚/中国近代思想史论.李泽厚著.三联书店2008.pdf` | 中国近代思想史论.李泽厚著.三联书店2008.pdf | 中国近代思想史论 | 李泽厚 | 否 |
| 349 | `Authors/李泽厚/华夏美学·美学四讲(增订本).李泽厚著.三联书店2008.pdf` | 华夏美学·美学四讲(增订本).李泽厚著.三联书店2008.pdf | 华夏美学·美学四讲(增订本) | 李泽厚 | 否 |
| 350 | `Authors/李泽厚/历史本体论·己卯五说(增订本).李泽厚著.三联书店2008.pdf` | 历史本体论·己卯五说(增订本).李泽厚著.三联书店2008.pdf | 历史本体论·己卯五说(增订本) | 李泽厚 | 否 |
| 351 | `Authors/李泽厚/批判哲学的批判_康德述评(修订第6版).李泽厚著.三联书店2007.pdf` | 批判哲学的批判_康德述评(修订第6版).李泽厚著.三联书店2007.pdf | 批判哲学的批判_康德述评(修订第6版) | 李泽厚 | 否 |
| 352 | `Authors/李泽厚/杂著集.李泽厚著.三联书店2008.pdf` | 杂著集.李泽厚著.三联书店2008.pdf | 杂著集 | 李泽厚 | 否 |
| 353 | `Authors/李泽厚/美的历程.李泽厚著.三联书店2009.pdf` | 美的历程.李泽厚著.三联书店2009.pdf | 美的历程 | 李泽厚 | 否 |
| 354 | `Authors/柄谷行人/(社会批判译丛) [日] 柄谷行人 著_ 中田友美 译 - 马克思，其可能性的中心-中央编译出版社 (2006).pdf` | (社会批判译丛) [日] 柄谷行人 著_ 中田友美 译 - 马克思，其可能性的中心-中央编译出版社 (2006).pdf | 马克思，其可能性的中心 | 柄谷行人 | 否 |
| 355 | `Authors/柄谷行人/作为隐喻的建筑_柄谷行人 著; 应杰 译.pdf` | 作为隐喻的建筑_柄谷行人 著; 应杰 译.pdf | 作为隐喻的建筑 | 柄谷行人 | 否 |
| 356 | `Authors/柄谷行人/日本现代文学的起源_[日]柄谷行人 著; 赵京华 译.pdf` | 日本现代文学的起源_[日]柄谷行人 著; 赵京华 译.pdf | 日本现代文学的起源 | 柄谷行人 | 否 |
| 357 | `Authors/柄谷行人/民族与美学_[日] 柄谷行人 著; 薛羽 译.pdf` | 民族与美学_[日] 柄谷行人 著; 薛羽 译.pdf | 民族与美学 | 柄谷行人 | 否 |
| 358 | `Authors/段义孚/Passing Strange and Wonderful_ Aesthetics, Nature, and Culture (Kodansha Globe)_Yi-Fu Tuan.pdf` | Passing Strange and Wonderful_ Aesthetics, Nature, and Culture (Kodansha Globe)_Yi-Fu Tuan.pdf | Passing Strange and Wonderful_ Aesthetics, Nature, and Culture | 段义孚 | 否 |
| 359 | `Authors/汤一介/汤一介集 (1).pdf` | 汤一介集 (1).pdf | 汤一介集 (1) | 汤一介 | 否 |
| 360 | `Authors/汤一介/汤一介集 (10).pdf` | 汤一介集 (10).pdf | 汤一介集 (10) | 汤一介 | 否 |
| 361 | `Authors/汤一介/汤一介集 (2).pdf` | 汤一介集 (2).pdf | 汤一介集 (2) | 汤一介 | 否 |
| 362 | `Authors/汤一介/汤一介集 (3).pdf` | 汤一介集 (3).pdf | 汤一介集 (3) | 汤一介 | 否 |
| 363 | `Authors/汤一介/汤一介集 (4).pdf` | 汤一介集 (4).pdf | 汤一介集 (4) | 汤一介 | 否 |
| 364 | `Authors/汤一介/汤一介集 (5).pdf` | 汤一介集 (5).pdf | 汤一介集 (5) | 汤一介 | 否 |
| 365 | `Authors/汤一介/汤一介集 (6).pdf` | 汤一介集 (6).pdf | 汤一介集 (6) | 汤一介 | 否 |
| 366 | `Authors/汤一介/汤一介集 (7).pdf` | 汤一介集 (7).pdf | 汤一介集 (7) | 汤一介 | 否 |
| 367 | `Authors/汤一介/汤一介集 (8).pdf` | 汤一介集 (8).pdf | 汤一介集 (8) | 汤一介 | 否 |
| 368 | `Authors/汤一介/汤一介集 (9).pdf` | 汤一介集 (9).pdf | 汤一介集 (9) | 汤一介 | 否 |
| 369 | `Authors/汪民安/城市文化读本_汪民安;_陈永国;_马海良.pdf` | 城市文化读本_汪民安;_陈永国;_马海良.pdf | 城市文化读本 | 汪民安 | 否 |
| 370 | `Authors/熊十力/熊十力全集·第4卷.pdf` | 熊十力全集·第4卷.pdf | 熊十力全集·第4卷 | 熊十力 | 否 |
| 371 | `Authors/熊十力/熊十力全集·第5卷.pdf` | 熊十力全集·第5卷.pdf | 熊十力全集·第5卷 | 熊十力 | 否 |
| 372 | `Authors/熊十力/熊十力全集·第6卷.pdf` | 熊十力全集·第6卷.pdf | 熊十力全集·第6卷 | 熊十力 | 否 |
| 373 | `Authors/熊十力/熊十力全集·第7卷.pdf` | 熊十力全集·第7卷.pdf | 熊十力全集·第7卷 | 熊十力 | 否 |
| 374 | `Authors/熊十力/熊十力全集·附卷上.pdf` | 熊十力全集·附卷上.pdf | 熊十力全集·附卷上 | 熊十力 | 否 |
| 375 | `Authors/蒋孔阳/美学原理（蒋孔阳） 0.pdf` | 美学原理（蒋孔阳） 0.pdf | 美学原理（蒋孔阳） 0 | 蒋孔阳 | 否 |
| 376 | `Authors/许煜/On the existence of digital objects-University of Minnesota Press (2017) - 许煜.pdf` | On the existence of digital objects-University of Minnesota Press (2017) - 许煜.pdf | On the existence of digital objects | 许煜 | 否 |
| 377 | `Authors/许煜/Question Concerning Technology in China An Essay in Cosmotechnics, The - 许煜.pdf` | Question Concerning Technology in China An Essay in Cosmotechnics, The - 许煜.pdf | Question Concerning Technology in China An Essay in Cosmotechnics, The | 许煜 | 否 |
| 378 | `Authors/郑元者/艺术之根 - 未知.pdf` | 艺术之根 - 未知.pdf | 艺术之根 | 郑元者 | 否 |
| 379 | `Authors/郑元者/艺术之根.pdf` | 艺术之根.pdf | 艺术之根 | 郑元者 | 否 |
| 380 | `Authors/郑振铎/《中国古代木刻画史略》_郑振铎 编著_上海书店出版社 2006年.pdf` | 《中国古代木刻画史略》_郑振铎 编著_上海书店出版社 2006年.pdf | 《中国古代木刻画史略》 | 郑振铎 | 否 |
| 381 | `Authors/郑振铎/《郑振铎全集 18 希腊神话与英雄传说》.pdf` | 《郑振铎全集 18 希腊神话与英雄传说》.pdf | 《郑振铎全集 18 希腊神话与英雄传说》 | 郑振铎 | 否 |
| 382 | `Authors/郑振铎/《郑振铎全集 19 俄国小说译文集》.pdf` | 《郑振铎全集 19 俄国小说译文集》.pdf | 《郑振铎全集 19 | 郑振铎 | 否 |
| 383 | `Authors/郑振铎/《郑振铎全集 20 太戈尔诗选及其他译文》.pdf` | 《郑振铎全集 20 太戈尔诗选及其他译文》.pdf | 《郑振铎全集 20 太戈尔诗选及其他译文》 | 郑振铎 | 否 |
| 384 | `Authors/郑振铎/《郑振铎全集 7_中国俗文学史》.pdf` | 《郑振铎全集 7_中国俗文学史》.pdf | 《郑振铎全集 7_中国俗文学史》 | 郑振铎 | 否 |
| 385 | `Authors/金在权/（Jaegwon Kim）Philosophy of mind.pdf` | （Jaegwon Kim）Philosophy of mind.pdf | （Jaegwon Kim）Philosophy of mind | 金在权 | 否 |
| 386 | `Authors/钱锺书/钱锺书集(三联版)·写在人生边上·人生边上的边上·石语.pdf` | 钱锺书集(三联版)·写在人生边上·人生边上的边上·石语.pdf | 钱锺书集(三联版)·写在人生边上·人生边上的边上·石语 | 钱锺书 | 否 |
| 387 | `Authors/韩炳哲/In the Swarm_ Digital Prospects_Byung-Chul Han, Erik Butler.pdf` | In the Swarm_ Digital Prospects_Byung-Chul Han, Erik Butler.pdf | In the Swarm_ Digital Prospects_Byung-Chul Han, Erik Butler | 韩炳哲 | 否 |
| 388 | `Authors/韩炳哲/The Agony of Eros_Byung-Chul Han, Erik Butler, Alain Badiou.pdf` | The Agony of Eros_Byung-Chul Han, Erik Butler, Alain Badiou.pdf | The Agony of Eros_Byung-Chul Han, Erik Butler, Alain Badiou | 韩炳哲 | 否 |
| 389 | `Authors/韩炳哲/韩炳哲作品系列（套装共9册）.epub` | 韩炳哲作品系列（套装共9册）.epub | 韩炳哲作品系列（套装共9册）.epub | 韩炳哲 | 否 |
| 390 | `Authors/马克思恩格斯/马克思恩格斯文集[第05卷] 资本论.pdf` | 马克思恩格斯文集[第05卷] 资本论.pdf | 马克思恩格斯文集[第05卷] 资本论 | 马克思恩格斯 | 否 |
| 391 | `Authors/高亨/高亨着作集林第07卷.墨经校诠.商君书注译.清华大学出版社.2004年1版1刷.pdf` | 高亨着作集林第07卷.墨经校诠.商君书注译.清华大学出版社.2004年1版1刷.pdf | 高亨着作集林第07卷.墨经校诠 | 高亨 | 否 |

## Zotero/（229 行）

| # | 原路径 | 文件名 | 标题 | 作者 | 父目录在盘 |
|---:|---|---|---|---|---|
| 1 | `Zotero/102679129-05-01-acc.pdf` | 102679129-05-01-acc.pdf | 102679129-05-01-acc |  | 是 |
| 2 | `Zotero/2021_History of blockchain-Blockchain 1.0 Currency.pdf` | 2021_History of blockchain-Blockchain 1.0 Currency.pdf | 2021_History of blockchain-Blockchain 1.0 Currency |  | 是 |
| 3 | `Zotero/Alguliyev_2021_Information security as a national security component.pdf` | Alguliyev_2021_Information security as a national security component.pdf | Information security as a national security component | Rasim M. Alguliyev\|Yadigar N. Imamverdiyev\|Rasim Sh. Mahmudov\|Ramiz M. Aliguliyev | 是 |
| 4 | `Zotero/Armaselu and Fickers_2024_Zoomland Exploring Scale in Digital History and Humanities.pdf` | Armaselu and Fickers_2024_Zoomland Exploring Scale in Digital History and Humanities.pdf | Zoomland Exploring Scale in Digital History and Humanities | Florentina Armaselu\|Andreas Fickers | 是 |
| 5 | `Zotero/Ault_2006_Wolfgang Tillmans in conjunction with the Exhibition Wolfgang Tillmans, which was co-organized by.pdf` | Ault_2006_Wolfgang Tillmans in conjunction with the Exhibition Wolfgang Tillmans, which was co-organized by.pdf | Wolfgang Tillmans in conjunction with the Exhibition Wolfgang Tillmans, which was co-organized by | Ault | 是 |
| 6 | `Zotero/Bhyravajjula et al._2025_so much depends  upon  a whitespace Why Whitespace Matters for Poets and LLMs.pdf` | Bhyravajjula et al._2025_so much depends  upon  a whitespace Why Whitespace Matters for Poets and LLMs.pdf | so much depends  upon  a whitespace Why Whitespace Matters for Poets and LLMs | Bhyravajjula | 是 |
| 7 | `Zotero/Bingham_2024_Education and Artificial Intelligence at the Scene of Writing A Derridean Consideration.pdf` | Bingham_2024_Education and Artificial Intelligence at the Scene of Writing A Derridean Consideration.pdf | Education and Artificial Intelligence at the Scene of Writing A Derridean Consideration | Bingham | 是 |
| 8 | `Zotero/Christou_2010_European Union security governance putting the ‘security’ back in.pdf` | Christou_2010_European Union security governance putting the ‘security’ back in.pdf | European Union security governance putting the ‘security’ back in | George Christou\|Stuart Croft\|Michela Ceccorulli\|Sonia Lucarelli | 是 |
| 9 | `Zotero/Dubberly and Pangaro_2009_On Modeling - What is conversation, and how can we design for it.pdf` | Dubberly and Pangaro_2009_On Modeling - What is conversation, and how can we design for it.pdf | On Modeling - What is conversation, and how can we design for it | Hugh Dubberly\|Paul Pangaro | 是 |
| 10 | `Zotero/Ecarnot_2015_Writing a scientific article A step-by-step guide for beginners.pdf` | Ecarnot_2015_Writing a scientific article A step-by-step guide for beginners.pdf | Writing a scientific article A step-by-step guide for beginners | F. Ecarnot\|M.-F. Seronde\|R. Chopard\|F. Schiele\|N. Meneveau | 是 |
| 11 | `Zotero/Eschert_2017_Gaining economic profit or losing cultural security Framing persuasive arguments for two types of c.` | Eschert_2017_Gaining economic profit or losing cultural security Framing persuasive arguments for two types of c.pdf | Gaining economic profit or losing cultural security Framing persuasive arguments for two types of c | Silke Eschert\|Michael Diehl\|René Ziegler | 是 |
| 12 | `Zotero/Fu_2022_Recalibrating, reconfiguring, and appropriating innovation a semantic network analysis of China’s m.pdf` | Fu_2022_Recalibrating, reconfiguring, and appropriating innovation a semantic network analysis of China’s m.pdf | Recalibrating, reconfiguring, and appropriating innovation a semantic network analysis of China’s m | Fu | 是 |
| 13 | `Zotero/GENHSI-2.pdf` | GENHSI-2.pdf | GENHSI-2 |  | 是 |
| 14 | `Zotero/Li_2015_Proceedings of 2015 2nd International Conference on Industrial Economics System and Industrial Secur.pdf` | Li_2015_Proceedings of 2015 2nd International Conference on Industrial Economics System and Industrial Secur.pdf | Proceedings of 2015 2nd International Conference on Industrial Economics System and Industrial Secur | Li | 是 |
| 15 | `Zotero/Liang_2026_Watermarking techniques for large language models a survey.pdf` | Liang_2026_Watermarking techniques for large language models a survey.pdf | Watermarking techniques for large language models a survey | Yuqing Liang\|Jiancheng Xiao\|Wensheng Gan\|Philip S. Yu | 是 |
| 16 | `Zotero/Los Angeles County Museum of Art_1971_A report on the Art and Technology Program of the Los Angeles County Museum` | Los Angeles County Museum of Art_1971_A report on the Art and Technology Program of the Los Angeles County Museum of Art | A report on the Art and Technology Program of the Los Angeles County Museum of Art, 1967-1971 | Los Angeles County Museum of Art | 是 |
| 17 | `Zotero/Misselhorn_2009_Imaginative Resistance. An Emotion-Based Account.pdf` | Misselhorn_2009_Imaginative Resistance. An Emotion-Based Account.pdf | Imaginative Resistance. An Emotion-Based Account | Catrin Misselhorn | 是 |
| 18 | `Zotero/Mohr_2013_Graphing the grammar of motives in National Security Strategies Cultural interpretation, automated.pdf` | Mohr_2013_Graphing the grammar of motives in National Security Strategies Cultural interpretation, automated.pdf | Graphing the grammar of motives in National Security Strategies Cultural interpretation, automated | Mohr | 是 |
| 19 | `Zotero/O'Sullivan_2024_The Bloomsbury handbook to the digital humanities.pdf` | O'Sullivan_2024_The Bloomsbury handbook to the digital humanities.pdf | The Bloomsbury handbook to the digital humanities | James O'Sullivan | 是 |
| 20 | `Zotero/Ostrovskaya_2015_The Discourse of the Russian Elite in the ERA of “Liquid” Modernity as a Problem of Ethnic, Soci` | Ostrovskaya_2015_The Discourse of the Russian Elite in the ERA of “Liquid” Modernity as a Problem of Ethnic, Social a.pd | The Discourse of the Russian Elite in the ERA of “Liquid” Modernity as a Problem of Ethnic, Social a | Tatyana A. Ostrovskaya\|Irina S. Karabulatova\|Zaineta R. Khachmafova\|Svetlana A. Lyausheva\|Gennady Vasilievich Osipov | 是 |
| 21 | `Zotero/Proletarianisation_b2_article_Final_.pdf` | Proletarianisation_b2_article_Final_.pdf | Proletarianisation_b2_article_Final_ |  | 是 |
| 22 | `Zotero/Rancière_WHAT MEDIUM CAN MEAN.pdf` | Rancière_WHAT MEDIUM CAN MEAN.pdf | Rancière_WHAT MEDIUM CAN MEAN |  | 是 |
| 23 | `Zotero/Rothbaum_2000_Attachment and culture Security in the United States and Japan..pdf` | Rothbaum_2000_Attachment and culture Security in the United States and Japan..pdf | Attachment and culture Security in the United States and Japan. | Fred Rothbaum\|John Weisz\|Martha Pott\|Kazuo Miyake\|Gilda Morelli | 是 |
| 24 | `Zotero/Sari_2021_Information security cultural differences among health care facilities in Indonesia.pdf` | Sari_2021_Information security cultural differences among health care facilities in Indonesia.pdf | Information security cultural differences among health care facilities in Indonesia | Puspita Kencana Sari\|Adhi Prasetio\|Putu Wuri Handayani\|Achmad Nizar Hidayanto\|Syaza Syauqina\|Eka Fuji Astuti\|Faris | 是 |
| 25 | `Zotero/Sattaka_2017_Agricultural extension services to foster production sustainability for food and cultural security o` | Sattaka_2017_Agricultural extension services to foster production sustainability for food and cultural security o.pdf | Agricultural extension services to foster production sustainability for food and cultural security o | Sattaka | 是 |
| 26 | `Zotero/Tomczak and Stachowiak_2015_Location Patterns And Location Factors In Cultural And Creative Industries1.pdf` | Tomczak and Stachowiak_2015_Location Patterns And Location Factors In Cultural And Creative Industries1.pdf | Location Patterns And Location Factors In Cultural And Creative Industries1 | Paulina Tomczak\|Krzysztof Stachowiak | 是 |
| 27 | `Zotero/greenberg_modernistPainting.pdf` | greenberg_modernistPainting.pdf | greenberg_modernistPainting |  | 是 |
| 28 | `Zotero/krauss_voyage-on-the-north-sea_cropped.pdf` | krauss_voyage-on-the-north-sea_cropped.pdf | krauss_voyage-on-the-north-sea_cropped |  | 是 |
| 29 | `Zotero/“饭圈”身份认同建构研究_刘念念.pdf` | “饭圈”身份认同建构研究_刘念念.pdf | “饭圈”身份认同建构研究_刘念念 |  | 是 |
| 30 | `Zotero/万_2018_“网络喷子”语言暴力的演变及引导思考.pdf` | 万_2018_“网络喷子”语言暴力的演变及引导思考.pdf | “网络喷子”语言暴力的演变及引导思考 | 万家凝 | 是 |
| 31 | `Zotero/严_2023_元宇宙图书馆的理论进展、实践问题与未来展望.pdf` | 严_2023_元宇宙图书馆的理论进展、实践问题与未来展望.pdf | 元宇宙图书馆的理论进展、实践问题与未来展望 | 严丹\|徐亚苹\|虞晨琳\|蔡迎春 | 是 |
| 32 | `Zotero/中国旅游产业融合发展机制理论及其应用研究_刘雪婷.pdf` | 中国旅游产业融合发展机制理论及其应用研究_刘雪婷.pdf | 中国旅游产业融合发展机制理论及其应用研究_刘雪婷 |  | 是 |
| 33 | `Zotero/习近平关于历史文化遗产保护利用重要论述研究_宁静.pdf` | 习近平关于历史文化遗产保护利用重要论述研究_宁静.pdf | 习近平关于历史文化遗产保护利用重要论述研究_宁静 |  | 是 |
| 34 | `Zotero/习近平文化强国观研究_蔡洁.pdf` | 习近平文化强国观研究_蔡洁.pdf | 习近平文化强国观研究_蔡洁 |  | 是 |
| 35 | `Zotero/于悠悠_社会网络视角下韧性艺术界的危机应对.pdf` | 于悠悠_社会网络视角下韧性艺术界的危机应对.pdf | 于悠悠_社会网络视角下韧性艺术界的危机应对 |  | 是 |
| 36 | `Zotero/付_2020_数字经济中国经济发展的新动能.pdf` | 付_2020_数字经济中国经济发展的新动能.pdf | 数字经济中国经济发展的新动能 | 付晓东 | 是 |
| 37 | `Zotero/代_2023_加速社会中的传播媒介与虚无主义解蔽及超越.pdf` | 代_2023_加速社会中的传播媒介与虚无主义解蔽及超越.pdf | 加速社会中的传播媒介与虚无主义解蔽及超越 | 代利刚 | 是 |
| 38 | `Zotero/佘_2021_智慧城市实践与智慧建设.pdf` | 佘_2021_智慧城市实践与智慧建设.pdf | 智慧城市实践与智慧建设 | 佘小强\|高声红\|李方政 | 是 |
| 39 | `Zotero/侍_2014_空间叙事方法缘起及在城市研究中的应用.pdf` | 侍_2014_空间叙事方法缘起及在城市研究中的应用.pdf | 空间叙事方法缘起及在城市研究中的应用 | 侍非\|高才驰\|孟璐\|蒋志杰 | 是 |
| 40 | `Zotero/侯_2009_基于城市意象变迁的城市旅游形象塑造研究——以江苏省扬州市为例.pdf` | 侯_2009_基于城市意象变迁的城市旅游形象塑造研究——以江苏省扬州市为例.pdf | 基于城市意象变迁的城市旅游形象塑造研究——以江苏省扬州市为例 | 侯兵\|黄震方\|尚正永 | 是 |
| 41 | `Zotero/侯_2014_基于语义文法的网络舆情精准分析方法研究.pdf` | 侯_2014_基于语义文法的网络舆情精准分析方法研究.pdf | 基于语义文法的网络舆情精准分析方法研究 | 侯圣峦\|刘磊\|曹存根 | 是 |
| 42 | `Zotero/党_2022_城市信息模型（CIM）赋能新型智慧城市发展综述.pdf` | 党_2022_城市信息模型（CIM）赋能新型智慧城市发展综述.pdf | 城市信息模型（CIM）赋能新型智慧城市发展综述 | 党安荣\|王飞飞\|曲葳\|韩雯雯\|田颖 | 是 |
| 43 | `Zotero/刘 and 马_2019_当今国外左翼加速主义思想研究.pdf` | 刘 and 马_2019_当今国外左翼加速主义思想研究.pdf | 当今国外左翼加速主义思想研究 | 刘秦民\|马希 | 是 |
| 44 | `Zotero/刘_2013_智慧伦敦未来智慧城市的范本.pdf` | 刘_2013_智慧伦敦未来智慧城市的范本.pdf | 智慧伦敦未来智慧城市的范本 | 刘长传 | 是 |
| 45 | `Zotero/刘_2015_中文网络文本的语义信息处理研究综述.pdf` | 刘_2015_中文网络文本的语义信息处理研究综述.pdf | 中文网络文本的语义信息处理研究综述 | 刘磊\|李壮\|张鑫\|吕帅 | 是 |
| 46 | `Zotero/刘_2016_北京东四片区历史街区的城市意象研究.pdf` | 刘_2016_北京东四片区历史街区的城市意象研究.pdf | 北京东四片区历史街区的城市意象研究 | 刘祎绯\|傅玮\|伍洋宇\|薛博文\|王思凡 | 是 |
| 47 | `Zotero/刘_2018_空间与媒介电影艺术中未来城市的诗意性表达.pdf` | 刘_2018_空间与媒介电影艺术中未来城市的诗意性表达.pdf | 空间与媒介电影艺术中未来城市的诗意性表达 | 刘晓希 | 是 |
| 48 | `Zotero/刘_2019_中国建设智慧社会的国外经验借鉴.pdf` | 刘_2019_中国建设智慧社会的国外经验借鉴.pdf | 中国建设智慧社会的国外经验借鉴 | 刘红芹\|汤志伟\|崔茜\|郭雨晖\|伍毓锋 | 是 |
| 49 | `Zotero/刘_2021_中国智慧城市试点政策对城市发展质量的影响——基于韧性发展的视角.pdf` | 刘_2021_中国智慧城市试点政策对城市发展质量的影响——基于韧性发展的视角.pdf | 中国智慧城市试点政策对城市发展质量的影响——基于韧性发展的视角 | 刘成杰\|胡钰苓\|李虹桥\|张娜 | 是 |
| 50 | `Zotero/刘_2022_总体国家安全观的历史唯物主义审视.pdf` | 刘_2022_总体国家安全观的历史唯物主义审视.pdf | 总体国家安全观的历史唯物主义审视 | 刘洪\|张天伟\|邓卓明 | 是 |
| 51 | `Zotero/刘_2022_日本智慧城市PPP平台的合作模式解读.pdf` | 刘_2022_日本智慧城市PPP平台的合作模式解读.pdf | 日本智慧城市PPP平台的合作模式解读 | 刘泉\|黄丁芳\|黄虎\|赖亚妮 | 是 |
| 52 | `Zotero/刘立宇_2010_新媒介与中国当代艺术变革.pdf` | 刘立宇_2010_新媒介与中国当代艺术变革.pdf | 新媒介与中国当代艺术变革 | 刘立宇 | 是 |
| 53 | `Zotero/单_2023_城市管理“一网统管”建设研究与思考.pdf` | 单_2023_城市管理“一网统管”建设研究与思考.pdf | 城市管理“一网统管”建设研究与思考 | 单斐\|陶瑜\|董正浩 | 是 |
| 54 | `Zotero/叶_2014_新加坡政府的立国智慧一场全民参与的智慧行动.pdf` | 叶_2014_新加坡政府的立国智慧一场全民参与的智慧行动.pdf | 新加坡政府的立国智慧一场全民参与的智慧行动 | 叶伟春 | 是 |
| 55 | `Zotero/向_2018_中国城市文化力发展评价指标体系研究.pdf` | 向_2018_中国城市文化力发展评价指标体系研究.pdf | 中国城市文化力发展评价指标体系研究 | 向勇\|白晓晴\|李尽沙 | 是 |
| 56 | `Zotero/吕_2021_新型智慧城市网络安全协同防护框架研究.pdf` | 吕_2021_新型智慧城市网络安全协同防护框架研究.pdf | 新型智慧城市网络安全协同防护框架研究 | 吕欣\|韩晓露\|郭晓萧\|张延强 | 是 |
| 57 | `Zotero/吴_2018_上海2035信息化引领“卓越的全球城市”建设.pdf` | 吴_2018_上海2035信息化引领“卓越的全球城市”建设.pdf | 上海2035信息化引领“卓越的全球城市”建设 | 吴勇毅 | 是 |
| 58 | `Zotero/吴_2022_智慧城市热潮下的“冷”思考学术笔谈.pdf` | 吴_2022_智慧城市热潮下的“冷”思考学术笔谈.pdf | 智慧城市热潮下的“冷”思考学术笔谈 | 吴志强\|王坚\|李德仁\|丁烈云\|党安荣\|甄峰\|杨滔\|刘合林\|杨俊宴\|龙瀛\|牛强\|赵渺希\|潘海啸\|王德\|汪光焘 | 是 |
| 59 | `Zotero/吴_2022_特色打造与可持续发展：南京乡村文旅IP塑造策略研究.pdf` | 吴_2022_特色打造与可持续发展：南京乡村文旅IP塑造策略研究.pdf | 特色打造与可持续发展：南京乡村文旅IP塑造策略研究 | 吴振一\|李倩\|黄诚淏\|陈冬 | 是 |
| 60 | `Zotero/周苑_2026_技术逻辑与艺术表征：新大众文艺视域中的网络微短剧 2.pdf` | 周苑_2026_技术逻辑与艺术表征：新大众文艺视域中的网络微短剧 2.pdf | 技术逻辑与艺术表征：新大众文艺视域中的网络微短剧 2 | 周苑 | 是 |
| 61 | `Zotero/国_2013_全球十大智慧城市巴黎绿色之城的建设经验.pdf` | 国_2013_全球十大智慧城市巴黎绿色之城的建设经验.pdf | 全球十大智慧城市巴黎绿色之城的建设经验 | 国智 | 是 |
| 62 | `Zotero/夏_2019_农业高质量发展：数字赋能与实现路径.pdf` | 夏_2019_农业高质量发展：数字赋能与实现路径.pdf | 农业高质量发展：数字赋能与实现路径 | 夏显力\|陈哲\|张慧利\|赵敏娟 | 是 |
| 63 | `Zotero/姚_2021_中国智慧城市研究的进展与展望.pdf` | 姚_2021_中国智慧城市研究的进展与展望.pdf | 中国智慧城市研究的进展与展望 | 姚冲\|甄峰\|席广亮 | 是 |
| 64 | `Zotero/姚_2022_智慧城市建设提高企业全要素生产率了吗.pdf` | 姚_2022_智慧城市建设提高企业全要素生产率了吗.pdf | 智慧城市建设提高企业全要素生产率了吗 | 姚圣文\|赵兰香\|张耀坤 | 是 |
| 65 | `Zotero/姜_2010_“具身化”知识、行动与时间性——从安迪·克拉克到吉尔·德勒兹.pdf` | 姜_2010_“具身化”知识、行动与时间性——从安迪·克拉克到吉尔·德勒兹.pdf | “具身化”知识、行动与时间性——从安迪·克拉克到吉尔·德勒兹 | 姜宇辉 | 是 |
| 66 | `Zotero/姜_2018_智慧·城市,基于国际视野下的思考.pdf` | 姜_2018_智慧·城市,基于国际视野下的思考.pdf | 智慧·城市,基于国际视野下的思考 | 姜鹏\|陈立群\|倪砼 | 是 |
| 67 | `Zotero/姜宇辉_2026_数字时代的主体性技术：操演、拥有或修补附视频.pdf` | 姜宇辉_2026_数字时代的主体性技术：操演、拥有或修补附视频.pdf | 数字时代的主体性技术：操演、拥有或修补附视频 | 姜宇辉 | 是 |
| 68 | `Zotero/孔_2020_网络媒介：重塑城市空间意象的新途径 1.pdf` | 孔_2020_网络媒介：重塑城市空间意象的新途径 1.pdf | 网络媒介：重塑城市空间意象的新途径 1 | 孔铎 | 是 |
| 69 | `Zotero/孙_2024_基于网络照片数据的工业遗产旅游地意象分析：以上海国际时尚中心为例.pdf` | 孙_2024_基于网络照片数据的工业遗产旅游地意象分析：以上海国际时尚中心为例.pdf | 基于网络照片数据的工业遗产旅游地意象分析：以上海国际时尚中心为例 | 孙德龙\|宋祎琳\|张抒妍\|陈心睿\|郝心\|王雯丽 | 是 |
| 70 | `Zotero/孟 and 吴_2022_重新审视“智慧城市”三个基本研究问题——基于英文文献系统性综述.pdf` | 孟 and 吴_2022_重新审视“智慧城市”三个基本研究问题——基于英文文献系统性综述.pdf | 重新审视“智慧城市”三个基本研究问题——基于英文文献系统性综述 | 孟凡坤\|吴湘玲 | 是 |
| 71 | `Zotero/宁小敏_2009_语义关联数据模型及其检索机制的研究.pdf` | 宁小敏_2009_语义关联数据模型及其检索机制的研究.pdf | 语义关联数据模型及其检索机制的研究 | 宁小敏 | 是 |
| 72 | `Zotero/安东尼奥·索马尼 and 易雨潇 - 2025 - “潜在空间”理论.pdf` | 安东尼奥·索马尼 and 易雨潇 - 2025 - “潜在空间”理论.pdf | 安东尼奥·索马尼 and 易雨潇 - 2025 - “潜在空间”理论 |  | 是 |
| 73 | `Zotero/山之辺ハサクィ_2026_シュトックハマー「フランシス・ポンジュと電気」を読む _20260627.pdf` | 山之辺ハサクィ_2026_シュトックハマー「フランシス・ポンジュと電気」を読む _20260627.pdf | シュトックハマー「フランシス・ポンジュと電気」を読む _20260627 | 山之辺ハサクィ | 是 |
| 74 | `Zotero/崔_2022_全球元宇宙产业化布局策略及启示.pdf` | 崔_2022_全球元宇宙产业化布局策略及启示.pdf | 全球元宇宙产业化布局策略及启示 | 崔冰\|马涛\|何颖 | 是 |
| 75 | `Zotero/巴_2016_基于关键词语义网络的领域主题演化分析方法研究.pdf` | 巴_2016_基于关键词语义网络的领域主题演化分析方法研究.pdf | 基于关键词语义网络的领域主题演化分析方法研究 | 巴志超\|杨子江\|朱世伟\|王蕾 | 是 |
| 76 | `Zotero/常_2022_中国5G新型智慧城市：内涵、问题与路径.pdf` | 常_2022_中国5G新型智慧城市：内涵、问题与路径.pdf | 中国5G新型智慧城市：内涵、问题与路径 | 常丁懿\|石娟\|郑鹏 | 是 |
| 77 | `Zotero/张_1997_技术、组织与文化的协同创新模式研究.pdf` | 张_1997_技术、组织与文化的协同创新模式研究.pdf | 技术、组织与文化的协同创新模式研究 | 张钢\|陈劲\|许庆瑞 | 是 |
| 78 | `Zotero/张_2013_角色神圣到世俗异化体育明星崇拜变迁的动因与理性审视.pdf` | 张_2013_角色神圣到世俗异化体育明星崇拜变迁的动因与理性审视.pdf | 角色神圣到世俗异化体育明星崇拜变迁的动因与理性审视 | 张本俊\|于鹏\|王东 | 是 |
| 79 | `Zotero/张_2014_智慧城市建设模式比较研究.pdf` | 张_2014_智慧城市建设模式比较研究.pdf | 智慧城市建设模式比较研究 | 张鸿\|范满航\|张超\|段禄峰 | 是 |
| 80 | `Zotero/张_2015_文化创新城市、空间与经济复兴.pdf` | 张_2015_文化创新城市、空间与经济复兴.pdf | 文化创新城市、空间与经济复兴 | 张晗 | 是 |
| 81 | `Zotero/张_2016_基于城市意象的旅游节庆品牌认知研究.pdf` | 张_2016_基于城市意象的旅游节庆品牌认知研究.pdf | 基于城市意象的旅游节庆品牌认知研究 | 张娇\|吴小根\|钟思琪 | 是 |
| 82 | `Zotero/张_2017_纽约科技创新要素集聚路径与启示借鉴.pdf` | 张_2017_纽约科技创新要素集聚路径与启示借鉴.pdf | 纽约科技创新要素集聚路径与启示借鉴 | 张微\|张玉娟\|金学慧 | 是 |
| 83 | `Zotero/张_2018_大数据背景下中国与新加坡智慧城市建设比较研究.pdf` | 张_2018_大数据背景下中国与新加坡智慧城市建设比较研究.pdf | 大数据背景下中国与新加坡智慧城市建设比较研究 | 张长亮\|韩雪雯\|李竟彤 | 是 |
| 84 | `Zotero/张_2020_基于文化产业企业网络视角的中国城市网络空间结构研究.pdf` | 张_2020_基于文化产业企业网络视角的中国城市网络空间结构研究.pdf | 基于文化产业企业网络视角的中国城市网络空间结构研究 | 张旭\|余方正\|徐良佳 | 是 |
| 85 | `Zotero/张_2020_基于社会网络分析研究中国创客政策.pdf` | 张_2020_基于社会网络分析研究中国创客政策.pdf | 基于社会网络分析研究中国创客政策 | 张丹\|李红亮\|方国威\|张佳\|王剑华 | 是 |
| 86 | `Zotero/张_2020_实践性与物质性粉丝文化研究的新理论框架.pdf` | 张_2020_实践性与物质性粉丝文化研究的新理论框架.pdf | 实践性与物质性粉丝文化研究的新理论框架 | 张磊 | 是 |
| 87 | `Zotero/张_2021_数字经济能否成为促进我国经济高质量发展的新动能.pdf` | 张_2021_数字经济能否成为促进我国经济高质量发展的新动能.pdf | 数字经济能否成为促进我国经济高质量发展的新动能 | 张腾\|蒋伏心\|韦朕韬 | 是 |
| 88 | `Zotero/张_2021_都市演艺集聚区的文化生产力研究基于百老汇、伦敦西区、上海演艺大世界的比较.pdf` | 张_2021_都市演艺集聚区的文化生产力研究基于百老汇、伦敦西区、上海演艺大世界的比较.pdf | 都市演艺集聚区的文化生产力研究基于百老汇、伦敦西区、上海演艺大世界的比较 | 张蕾 | 是 |
| 89 | `Zotero/张_2023_回应AIGC的信息资源管理学人思考.pdf` | 张_2023_回应AIGC的信息资源管理学人思考.pdf | 回应AIGC的信息资源管理学人思考 | 张智雄\|曾建勋\|夏翠娟\|王东波\|李白杨\|蔡迎春 | 是 |
| 90 | `Zotero/张晶_2011_艺术媒介论.pdf` | 张晶_2011_艺术媒介论.pdf | 艺术媒介论 | 张晶 | 是 |
| 91 | `Zotero/张绍江,辛萍,盖小飞,周玉丽_1995_关于劳动力商品属性的探讨.pdf` | 张绍江,辛萍,盖小飞,周玉丽_1995_关于劳动力商品属性的探讨.pdf | 关于劳动力商品属性的探讨 | 张绍江\|辛萍\|盖小飞\|周玉丽 | 是 |
| 92 | `Zotero/徐 and 汪_2020_社会语境变迁下的中国科学家媒介形象研究——基于《人民日报》(1949—2019)的语义网络分析.pdf` | 徐 and 汪_2020_社会语境变迁下的中国科学家媒介形象研究——基于《人民日报》(1949—2019)的语义网络分析.pdf | 社会语境变迁下的中国科学家媒介形象研究——基于《人民日报》(1949—2019)的语义网络分析 | 徐素田\|汪凯 | 是 |
| 93 | `Zotero/徐_2012_旅游意象图基于游客感知的旅游景区规划新设想.pdf` | 徐_2012_旅游意象图基于游客感知的旅游景区规划新设想.pdf | 旅游意象图基于游客感知的旅游景区规划新设想 | 徐美\|刘春腊\|陈建设\|刘沛林 | 是 |
| 94 | `Zotero/徐_2020_想象的城市与城市的想象空间视阈下城市文化产业的价值探讨.pdf` | 徐_2020_想象的城市与城市的想象空间视阈下城市文化产业的价值探讨.pdf | 想象的城市与城市的想象空间视阈下城市文化产业的价值探讨 | 徐丹丹 | 是 |
| 95 | `Zotero/徐_2023_理解平台资本主义：概念生成、现实表征与公共性隐忧.pdf` | 徐_2023_理解平台资本主义：概念生成、现实表征与公共性隐忧.pdf | 理解平台资本主义：概念生成、现实表征与公共性隐忧 | 徐敬宏\|袁宇航\|沈晓霞 | 是 |
| 96 | `Zotero/徐露 and 马睿 - 2025 - 表面转向：当代批评与知识生成的新范式.pdf` | 徐露 and 马睿 - 2025 - 表面转向：当代批评与知识生成的新范式.pdf | 徐露 and 马睿 - 2025 - 表面转向：当代批评与知识生成的新范式 |  | 是 |
| 97 | `Zotero/戴_2023_基于Unity的在线地方特色卡牌游戏设计.pdf` | 戴_2023_基于Unity的在线地方特色卡牌游戏设计.pdf | 基于Unity的在线地方特色卡牌游戏设计 | 戴培山\|马凌尧\|范敏 | 是 |
| 98 | `Zotero/房 and 王_2017_智慧城市建设中外模式比较与文化产业创生逻辑.pdf` | 房 and 王_2017_智慧城市建设中外模式比较与文化产业创生逻辑.pdf | 智慧城市建设中外模式比较与文化产业创生逻辑 | 房勇\|王广振 | 是 |
| 99 | `Zotero/方_2022_多中心治理理论视角下乡村文化治理.pdf` | 方_2022_多中心治理理论视角下乡村文化治理.pdf | 多中心治理理论视角下乡村文化治理 | 方军\|张国祥\|任志安 | 是 |
| 100 | `Zotero/无面者归来：数字时代的面容溃散及其抵抗_施畅.pdf` | 无面者归来：数字时代的面容溃散及其抵抗_施畅.pdf | 无面者归来：数字时代的面容溃散及其抵抗_施畅 |  | 是 |
| 101 | `Zotero/智慧城市信息安全风险评估指标体系构建_高凯.pdf` | 智慧城市信息安全风险评估指标体系构建_高凯.pdf | 智慧城市信息安全风险评估指标体系构建_高凯 |  | 是 |
| 102 | `Zotero/曹_2017_基于网络照片数据的城市意象研究——以中国24个主要城市为例.pdf` | 曹_2017_基于网络照片数据的城市意象研究——以中国24个主要城市为例.pdf | 基于网络照片数据的城市意象研究——以中国24个主要城市为例 | 曹越皓\|龙瀛\|杨培峰 | 是 |
| 103 | `Zotero/曾_2021_情感慰藉、柔性社交、价值变现青年亚文化视域下的盲盒潮玩.pdf` | 曾_2021_情感慰藉、柔性社交、价值变现青年亚文化视域下的盲盒潮玩.pdf | 情感慰藉、柔性社交、价值变现青年亚文化视域下的盲盒潮玩 | 曾昕 | 是 |
| 104 | `Zotero/朱 and 韩_2017_拟态亲密关系一项关于养成系偶像粉丝社群的新观察——以TFboys个案为例.pdf` | 朱 and 韩_2017_拟态亲密关系一项关于养成系偶像粉丝社群的新观察——以TFboys个案为例.pdf | 拟态亲密关系一项关于养成系偶像粉丝社群的新观察——以TFboys个案为例 | 朱丽丽\|韩怡辰 | 是 |
| 105 | `Zotero/朱竑 and 高权_2015_西方地理学“情感转向”与情感地理学研究述评.pdf` | 朱竑 and 高权_2015_西方地理学“情感转向”与情感地理学研究述评.pdf | 西方地理学“情感转向”与情感地理学研究述评 | 朱竑\|高权 | 是 |
| 106 | `Zotero/朱迪·瓦克曼的女性主义社会加速理论研究_周琪.pdf` | 朱迪·瓦克曼的女性主义社会加速理论研究_周琪.pdf | 朱迪·瓦克曼的女性主义社会加速理论研究_周琪 |  | 是 |
| 107 | `Zotero/李 and 彭_2008_文化与创意产业集群的研究谱系和前沿走向文化生态隐喻.pdf` | 李 and 彭_2008_文化与创意产业集群的研究谱系和前沿走向文化生态隐喻.pdf | 文化与创意产业集群的研究谱系和前沿走向文化生态隐喻 | 李蕾蕾\|彭素英 | 是 |
| 108 | `Zotero/李_2010_从地理学视角看城市历史文化景观集体记忆的研究.pdf` | 李_2010_从地理学视角看城市历史文化景观集体记忆的研究.pdf | 从地理学视角看城市历史文化景观集体记忆的研究 | 李凡\|朱竑\|黄维 | 是 |
| 109 | `Zotero/李_2018_城市文化产业生态位测度及空间网络效应.pdf` | 李_2018_城市文化产业生态位测度及空间网络效应.pdf | 城市文化产业生态位测度及空间网络效应 | 李建春\|袁文华\|吴美玉\|邱路 | 是 |
| 110 | `Zotero/李_2019_精神负载母职实践过程中的性别关系再生产.pdf` | 李_2019_精神负载母职实践过程中的性别关系再生产.pdf | 精神负载母职实践过程中的性别关系再生产 | 李芳英 | 是 |
| 111 | `Zotero/李_2021_文化产业成长对城市空间形态演变的影响——以西安市核心区为例.pdf` | 李_2021_文化产业成长对城市空间形态演变的影响——以西安市核心区为例.pdf | 文化产业成长对城市空间形态演变的影响——以西安市核心区为例 | 李留通\|张森森\|赵新正\|权东计\|罗伊 | 是 |
| 112 | `Zotero/李_2021_智慧城市网络安全保障研究进展.pdf` | 李_2021_智慧城市网络安全保障研究进展.pdf | 智慧城市网络安全保障研究进展 | 李阳\|郭晓萧\|张延强 | 是 |
| 113 | `Zotero/李_2021_消费符号下城市新旧电影院空间生产及功能演化——以西安市为例.pdf` | 李_2021_消费符号下城市新旧电影院空间生产及功能演化——以西安市为例.pdf | 消费符号下城市新旧电影院空间生产及功能演化——以西安市为例 | 李修远\|杨晓俊\|孔资淇\|雷蕾 | 是 |
| 114 | `Zotero/李_2021_跨区域合作智慧城市拓展新方向.pdf` | 李_2021_跨区域合作智慧城市拓展新方向.pdf | 跨区域合作智慧城市拓展新方向 | 李农 | 是 |
| 115 | `Zotero/李_2023_总体国家安全观视域下我国网络文化安全法治保障研究.pdf` | 李_2023_总体国家安全观视域下我国网络文化安全法治保障研究.pdf | 总体国家安全观视域下我国网络文化安全法治保障研究 | 李天英\|赵雅芳\|朱梦迪\|张彩红 | 是 |
| 116 | `Zotero/杨_2012_“智慧城市”的建设策略对全球优秀实践的分析与思考.pdf` | 杨_2012_“智慧城市”的建设策略对全球优秀实践的分析与思考.pdf | “智慧城市”的建设策略对全球优秀实践的分析与思考 | 杨红艳 | 是 |
| 117 | `Zotero/杨_2019_技术女性主义视角下的时间经验——瓦克曼《时间逼迫数字资本主义中生命的加速》解读.pdf` | 杨_2019_技术女性主义视角下的时间经验——瓦克曼《时间逼迫数字资本主义中生命的加速》解读.pdf | 技术女性主义视角下的时间经验——瓦克曼《时间逼迫数字资本主义中生命的加速》解读 | 杨立学 | 是 |
| 118 | `Zotero/杨_2020_基于网络图景的重庆主城空间意象研究.pdf` | 杨_2020_基于网络图景的重庆主城空间意象研究.pdf | 基于网络图景的重庆主城空间意象研究 | 杨迪\|高银宝\|赵潇欣 | 是 |
| 119 | `Zotero/杨_2022_人工智能文化产业创新逻辑及其价值风险与治理路径.pdf` | 杨_2022_人工智能文化产业创新逻辑及其价值风险与治理路径.pdf | 人工智能文化产业创新逻辑及其价值风险与治理路径 | 杨旦修\|王茜芮\|吕冠霖 | 是 |
| 120 | `Zotero/杨光影 and 马睿 - 2024 - 论科技艺术研究的“新媒介”之困.pdf` | 杨光影 and 马睿 - 2024 - 论科技艺术研究的“新媒介”之困.pdf | 杨光影 and 马睿 - 2024 - 论科技艺术研究的“新媒介”之困 |  | 是 |
| 121 | `Zotero/杨光影 and 马睿 - 2025 - 感知的重审：新媒体艺术家的媒介研究.pdf` | 杨光影 and 马睿 - 2025 - 感知的重审：新媒体艺术家的媒介研究.pdf | 杨光影 and 马睿 - 2025 - 感知的重审：新媒体艺术家的媒介研究 |  | 是 |
| 122 | `Zotero/林_2016_产业融合与空间融合文化产业融合发展的思维创新.pdf` | 林_2016_产业融合与空间融合文化产业融合发展的思维创新.pdf | 产业融合与空间融合文化产业融合发展的思维创新 | 林秀琴 | 是 |
| 123 | `Zotero/栗_2021_基于WOS和CNKI可视化的智慧城市研究知识图谱分析.pdf` | 栗_2021_基于WOS和CNKI可视化的智慧城市研究知识图谱分析.pdf | 基于WOS和CNKI可视化的智慧城市研究知识图谱分析 | 栗俊杰\|王闻珑\|刘邦凡 | 是 |
| 124 | `Zotero/梁_2023_数字技术下“电影-元宇宙”工业模型搭建逻辑支撑与实践校验.pdf` | 梁_2023_数字技术下“电影-元宇宙”工业模型搭建逻辑支撑与实践校验.pdf | 数字技术下“电影-元宇宙”工业模型搭建逻辑支撑与实践校验 | 梁思远 | 是 |
| 125 | `Zotero/武_2022_智慧城市建设对绿色技术创新的影响——基于地级市面板数据的实证研究.pdf` | 武_2022_智慧城市建设对绿色技术创新的影响——基于地级市面板数据的实证研究.pdf | 智慧城市建设对绿色技术创新的影响——基于地级市面板数据的实证研究 | 武力超\|李惟简\|陈丽玲\|李嘉欣 | 是 |
| 126 | `Zotero/段 and 陈_2020_文化伦理人工智能的价值与悖论.pdf` | 段 and 陈_2020_文化伦理人工智能的价值与悖论.pdf | 文化伦理人工智能的价值与悖论 | 段峰峰\|陈星豫 | 是 |
| 127 | `Zotero/殷_2014_“大流转”中国都市文学的梦想与纠结.pdf` | 殷_2014_“大流转”中国都市文学的梦想与纠结.pdf | “大流转”中国都市文学的梦想与纠结 | 殷国明 | 是 |
| 128 | `Zotero/殷_2019_“思想战”关于“文化战争”的前奏和先导——20世纪中国文学批评百年记忆.pdf` | 殷_2019_“思想战”关于“文化战争”的前奏和先导——20世纪中国文学批评百年记忆.pdf | “思想战”关于“文化战争”的前奏和先导——20世纪中国文学批评百年记忆 | 殷国明 | 是 |
| 129 | `Zotero/殷_2024_作为媒介的城市数字媒介技术在城市品牌传播中的应用分析——以南京文旅夜游品牌为例.pdf` | 殷_2024_作为媒介的城市数字媒介技术在城市品牌传播中的应用分析——以南京文旅夜游品牌为例.pdf | 作为媒介的城市数字媒介技术在城市品牌传播中的应用分析——以南京文旅夜游品牌为例 | 殷乐希 | 是 |
| 130 | `Zotero/毕_2022_沉浸式旅游演艺：内涵、特征及发展动力.pdf` | 毕_2022_沉浸式旅游演艺：内涵、特征及发展动力.pdf | 沉浸式旅游演艺：内涵、特征及发展动力 | 毕剑\|贾苏萍\|周成 | 是 |
| 131 | `Zotero/毛哲涵_王鹏_2023_应用元宇宙促进城市空间规划变革研究.pdf` | 毛哲涵_王鹏_2023_应用元宇宙促进城市空间规划变革研究.pdf | 应用元宇宙促进城市空间规划变革研究 | 毛哲涵\|王鹏 | 是 |
| 132 | `Zotero/江_2024_我国城市夜间文旅消费集聚区网络口碑空间格局及影响因素分析.pdf` | 江_2024_我国城市夜间文旅消费集聚区网络口碑空间格局及影响因素分析.pdf | 我国城市夜间文旅消费集聚区网络口碑空间格局及影响因素分析 | 江丽珍\|冯兴华\|傅钰\|李建新\|年柏翰 | 是 |
| 133 | `Zotero/汪_2024_城市空间生产与形象建构中的身体悖论 1.pdf` | 汪_2024_城市空间生产与形象建构中的身体悖论 1.pdf | 城市空间生产与形象建构中的身体悖论 1 | 汪金汉 | 是 |
| 134 | `Zotero/沈_2014_辣妈个体化进程中母职与女权.pdf` | 沈_2014_辣妈个体化进程中母职与女权.pdf | 辣妈个体化进程中母职与女权 | 沈奕斐 | 是 |
| 135 | `Zotero/沈_2020_智慧城市建设中公民参与的实践探索与启示.pdf` | 沈_2020_智慧城市建设中公民参与的实践探索与启示.pdf | 智慧城市建设中公民参与的实践探索与启示 | 沈心怡\|李德智\|张帆 | 是 |
| 136 | `Zotero/涂_2020_伦敦智慧城市用户信息安全保障机制及其对我国的启示.pdf` | 涂_2020_伦敦智慧城市用户信息安全保障机制及其对我国的启示.pdf | 伦敦智慧城市用户信息安全保障机制及其对我国的启示 | 涂萌\|何培育\|陈博 | 是 |
| 137 | `Zotero/温_2022_数字化背景下智慧城市的治理效应及治理过程研究.pdf` | 温_2022_数字化背景下智慧城市的治理效应及治理过程研究.pdf | 数字化背景下智慧城市的治理效应及治理过程研究 | 温雅婷\|余江\|洪志生\|陈凤 | 是 |
| 138 | `Zotero/焦_2020_数字经济赋能制造业转型从价值重塑到价值创造.pdf` | 焦_2020_数字经济赋能制造业转型从价值重塑到价值创造.pdf | 数字经济赋能制造业转型从价值重塑到价值创造 | 焦勇 | 是 |
| 139 | `Zotero/王 and 刘_2021_长三角文化产业集群发展实践成效、现实困境与战略路径.pdf` | 王 and 刘_2021_长三角文化产业集群发展实践成效、现实困境与战略路径.pdf | 长三角文化产业集群发展实践成效、现实困境与战略路径 | 王文姬\|刘柏阳 | 是 |
| 140 | `Zotero/王_2013_国内外智慧城市理论研究与实践思考.pdf` | 王_2013_国内外智慧城市理论研究与实践思考.pdf | 国内外智慧城市理论研究与实践思考 | 王广斌\|张雷\|刘洪磊 | 是 |
| 141 | `Zotero/王_2013_艺术介入空间生产城市闲置工业空间的旅游开发.pdf` | 王_2013_艺术介入空间生产城市闲置工业空间的旅游开发.pdf | 艺术介入空间生产城市闲置工业空间的旅游开发 | 王娟\|张广海\|张凌云 | 是 |
| 142 | `Zotero/王_2015_一些国家智慧城市建设的做法及其重要启示(上).pdf` | 王_2015_一些国家智慧城市建设的做法及其重要启示(上).pdf | 一些国家智慧城市建设的做法及其重要启示(上) | 王德春 | 是 |
| 143 | `Zotero/王_2015_一些国家智慧城市建设的做法及其重要启示(下).pdf` | 王_2015_一些国家智慧城市建设的做法及其重要启示(下).pdf | 一些国家智慧城市建设的做法及其重要启示(下) | 王德春 | 是 |
| 144 | `Zotero/王_2017_以“智能”实现城市公平——美国纽约的智慧城市创建行动及启示.pdf` | 王_2017_以“智能”实现城市公平——美国纽约的智慧城市创建行动及启示.pdf | 以“智能”实现城市公平——美国纽约的智慧城市创建行动及启示 | 王胤瑜\|田大江\|李颖玥 | 是 |
| 145 | `Zotero/王_2017_浅议英国格林威治智慧城市战略对我国的启示.pdf` | 王_2017_浅议英国格林威治智慧城市战略对我国的启示.pdf | 浅议英国格林威治智慧城市战略对我国的启示 | 王元汉\|田大江\|刘朝晖 | 是 |
| 146 | `Zotero/王_2019_智慧城市发展的经验分析与趋势展望.pdf` | 王_2019_智慧城市发展的经验分析与趋势展望.pdf | 智慧城市发展的经验分析与趋势展望 | 王哲\|郑子亨\|周斌\|赵燕 | 是 |
| 147 | `Zotero/王_2020_城市智慧化理念演进从工具理性到价值理性——基于上海智慧城市十年发展考察.pdf` | 王_2020_城市智慧化理念演进从工具理性到价值理性——基于上海智慧城市十年发展考察.pdf | 城市智慧化理念演进从工具理性到价值理性——基于上海智慧城市十年发展考察 | 王操 | 是 |
| 148 | `Zotero/王_2020_数字文化产业中政府角色定位欧美国家经验与中国对策.pdf` | 王_2020_数字文化产业中政府角色定位欧美国家经验与中国对策.pdf | 数字文化产业中政府角色定位欧美国家经验与中国对策 | 王相华 | 是 |
| 149 | `Zotero/王_2022_公共文化治理理念、模式与实践路向.pdf` | 王_2022_公共文化治理理念、模式与实践路向.pdf | 公共文化治理理念、模式与实践路向 | 王滢涛 | 是 |
| 150 | `Zotero/王_2022_西安市文化娱乐业空间结构图谱及其形成机理.pdf` | 王_2022_西安市文化娱乐业空间结构图谱及其形成机理.pdf | 西安市文化娱乐业空间结构图谱及其形成机理 | 王怡婕\|薛东前\|宋永永\|马蓓蓓 | 是 |
| 151 | `Zotero/王千桂_2015_走向“再生现实”的新媒体戏剧  基于新媒体戏剧演艺的舞台美术现状解读与分析.pdf` | 王千桂_2015_走向“再生现实”的新媒体戏剧  基于新媒体戏剧演艺的舞台美术现状解读与分析.pdf | 走向“再生现实”的新媒体戏剧  基于新媒体戏剧演艺的舞台美术现状解读与分析 | 王千桂 | 是 |
| 152 | `Zotero/王睿琦_2016_从形式到结构——作为结构主义文本的《恋人絮语》.pdf` | 王睿琦_2016_从形式到结构——作为结构主义文本的《恋人絮语》.pdf | 从形式到结构——作为结构主义文本的《恋人絮语》 | 王睿琦 | 是 |
| 153 | `Zotero/田_2022_智慧城市信息安全保障体系研究.pdf` | 田_2022_智慧城市信息安全保障体系研究.pdf | 智慧城市信息安全保障体系研究 | 田珍\|史龑东\|李仪涵 | 是 |
| 154 | `Zotero/由_2021_文化企业承担社会责任会提高其融资效率吗.pdf` | 由_2021_文化企业承担社会责任会提高其融资效率吗.pdf | 文化企业承担社会责任会提高其融资效率吗 | 由婉珍\|史占中\|陈能军 | 是 |
| 155 | `Zotero/盖小飞 and 解学芳_2015_我国文化创意产业制度变迁与困境.pdf` | 盖小飞 and 解学芳_2015_我国文化创意产业制度变迁与困境.pdf | 我国文化创意产业制度变迁与困境 | 盖小飞\|解学芳 | 是 |
| 156 | `Zotero/盖小飞 and 解学芳_2015_文化创意产业制度环境的优化策略.pdf` | 盖小飞 and 解学芳_2015_文化创意产业制度环境的优化策略.pdf | 文化创意产业制度环境的优化策略 | 盖小飞\|解学芳 | 是 |
| 157 | `Zotero/盖小飞 and 解学芳_2016_新能源汽车产业与大众传播媒介的互动关系研究.pdf` | 盖小飞 and 解学芳_2016_新能源汽车产业与大众传播媒介的互动关系研究.pdf | 新能源汽车产业与大众传播媒介的互动关系研究 | 盖小飞\|解学芳 | 是 |
| 158 | `Zotero/盖小飞_2016_“犬儒主义”与收视率——布尔迪厄《关于电视》解析.pdf` | 盖小飞_2016_“犬儒主义”与收视率——布尔迪厄《关于电视》解析.pdf | “犬儒主义”与收视率——布尔迪厄《关于电视》解析 | 盖小飞 | 是 |
| 159 | `Zotero/盖小飞_2016_哈尼梯田地区文化消费现状研究.pdf` | 盖小飞_2016_哈尼梯田地区文化消费现状研究.pdf | 哈尼梯田地区文化消费现状研究 | 盖小飞 | 是 |
| 160 | `Zotero/盖小飞_2016_哈尼梯田文化旅游开发现状初探——以“长街宴”和“梯田红米”为例.pdf` | 盖小飞_2016_哈尼梯田文化旅游开发现状初探——以“长街宴”和“梯田红米”为例.pdf | 哈尼梯田文化旅游开发现状初探——以“长街宴”和“梯田红米”为例 | 盖小飞 | 是 |
| 161 | `Zotero/盖小飞_2016_真人秀节目《奔跑吧兄弟》版权引进研究综述.pdf` | 盖小飞_2016_真人秀节目《奔跑吧兄弟》版权引进研究综述.pdf | 真人秀节目《奔跑吧兄弟》版权引进研究综述 | 盖小飞 | 是 |
| 162 | `Zotero/盖小飞_2016_论消费主义中选择的自由与困惑.pdf` | 盖小飞_2016_论消费主义中选择的自由与困惑.pdf | 论消费主义中选择的自由与困惑 | 盖小飞 | 是 |
| 163 | `Zotero/秦_2013_用“智慧规划”助推上海智慧城市建设刍议.pdf` | 秦_2013_用“智慧规划”助推上海智慧城市建设刍议.pdf | 用“智慧规划”助推上海智慧城市建设刍议 | 秦战\|杨心丽\|王颖莹 | 是 |
| 164 | `Zotero/罗_2017_从“赋权”到“赋能”的企业组织结构演进——基于韩都衣舍案例的研究.pdf` | 罗_2017_从“赋权”到“赋能”的企业组织结构演进——基于韩都衣舍案例的研究.pdf | 从“赋权”到“赋能”的企业组织结构演进——基于韩都衣舍案例的研究 | 罗仲伟\|李先军\|宋翔\|李亚光 | 是 |
| 165 | `Zotero/罗_2022_文化产业数字化的创新衍进模式研究.pdf` | 罗_2022_文化产业数字化的创新衍进模式研究.pdf | 文化产业数字化的创新衍进模式研究 | 罗仕鉴\|张德寅\|沈诚仪\|卢杨 | 是 |
| 166 | `Zotero/翟_2017_旅游空间生产语境下的乡村文化景观失忆与重构研究——以成都市红砂村乡村旅游发展为例.pdf` | 翟_2017_旅游空间生产语境下的乡村文化景观失忆与重构研究——以成都市红砂村乡村旅游发展为例.pdf | 旅游空间生产语境下的乡村文化景观失忆与重构研究——以成都市红砂村乡村旅游发展为例 | 翟向坤\|郭凌\|张晓\|陈爱莉 | 是 |
| 167 | `Zotero/胡 and 姬_2017_网络文化安全概念、规范与趋势.pdf` | 胡 and 姬_2017_网络文化安全概念、规范与趋势.pdf | 网络文化安全概念、规范与趋势 | 胡正荣\|姬德强 | 是 |
| 168 | `Zotero/胡 and 罗_2016_城市形象塑造基于文化产业视角.pdf` | 胡 and 罗_2016_城市形象塑造基于文化产业视角.pdf | 城市形象塑造基于文化产业视角 | 胡攀\|罗锐华 | 是 |
| 169 | `Zotero/胡_2000_国家文化安全经济全球化背景下中国文化产业发展策论.pdf` | 胡_2000_国家文化安全经济全球化背景下中国文化产业发展策论.pdf | 国家文化安全经济全球化背景下中国文化产业发展策论 | 胡惠林 | 是 |
| 170 | `Zotero/臧 and 解_2019_人工智能时代文化产业主流价值传播重塑与建构.pdf` | 臧 and 解_2019_人工智能时代文化产业主流价值传播重塑与建构.pdf | 人工智能时代文化产业主流价值传播重塑与建构 | 臧志彭\|解学芳 | 是 |
| 171 | `Zotero/范 and 张_2016_城市文化空间传承与创意的映射.pdf` | 范 and 张_2016_城市文化空间传承与创意的映射.pdf | 城市文化空间传承与创意的映射 | 范红\|张皓 | 是 |
| 172 | `Zotero/茅明睿 et al_2023_基于游戏和元宇宙技术的城市研究与空间干预探索.pdf` | 茅明睿 et al_2023_基于游戏和元宇宙技术的城市研究与空间干预探索.pdf | 基于游戏和元宇宙技术的城市研究与空间干预探索 | 茅明睿 | 是 |
| 173 | `Zotero/葛 and 吴_2021_数字经济赋能经济高质量发展理论机制与经验证据.pdf` | 葛 and 吴_2021_数字经济赋能经济高质量发展理论机制与经验证据.pdf | 数字经济赋能经济高质量发展理论机制与经验证据 | 葛和平\|吴福象 | 是 |
| 174 | `Zotero/董_2022_基于分阶段创新活动需求的城市创新集聚区类型解析与构建引导.pdf` | 董_2022_基于分阶段创新活动需求的城市创新集聚区类型解析与构建引导.pdf | 基于分阶段创新活动需求的城市创新集聚区类型解析与构建引导 | 董琦\|于凯\|刘航\|耿煜周 | 是 |
| 175 | `Zotero/蔡 and 卞_2018_青蛙游戏代际冲突下的青年成长.pdf` | 蔡 and 卞_2018_青蛙游戏代际冲突下的青年成长.pdf | 青蛙游戏代际冲突下的青年成长 | 蔡骐\|卞寒月 | 是 |
| 176 | `Zotero/薛_2022_城市历史街区公共空间更新的新文旅视角与新赋能机制——以哈尔滨市为例.pdf` | 薛_2022_城市历史街区公共空间更新的新文旅视角与新赋能机制——以哈尔滨市为例.pdf | 城市历史街区公共空间更新的新文旅视角与新赋能机制——以哈尔滨市为例 | 薛名辉\|万子祎\|李佳 | 是 |
| 177 | `Zotero/袁_2020_基于“义利”分析框架的文化企业社会责任与经济绩效关系研究——来自72家文化产业上市公司的实证分析.pdf` | 袁_2020_基于“义利”分析框架的文化企业社会责任与经济绩效关系研究——来自72家文化产业上市公司的实证分析.pdf | 基于“义利”分析框架的文化企业社会责任与经济绩效关系研究——来自72家文化产业上市公司的实证分析 | 袁文华\|秦晓楠\|吴美玉 | 是 |
| 178 | `Zotero/袁_2023_粉丝经济视域下虚拟偶像的发展策略.pdf` | 袁_2023_粉丝经济视域下虚拟偶像的发展策略.pdf | 粉丝经济视域下虚拟偶像的发展策略 | 袁华\|康苗\|刘志鹏 | 是 |
| 179 | `Zotero/解 and 张_2020_AI赋能人工智能与媒体产业链重构.pdf` | 解 and 张_2020_AI赋能人工智能与媒体产业链重构.pdf | AI赋能人工智能与媒体产业链重构 | 解学芳\|张佳琪 | 是 |
| 180 | `Zotero/解 and 高_2022_“智能+”时代AI文化科技伦理反思创新边界与伦理准则.pdf` | 解 and 高_2022_“智能+”时代AI文化科技伦理反思创新边界与伦理准则.pdf | “智能+”时代AI文化科技伦理反思创新边界与伦理准则 | 解学芳\|高嘉琪 | 是 |
| 181 | `Zotero/解_2011_网络文化产业管理体制改革技术创新驱动的缺位与突破.pdf` | 解_2011_网络文化产业管理体制改革技术创新驱动的缺位与突破.pdf | 网络文化产业管理体制改革技术创新驱动的缺位与突破 | 解学芳 | 是 |
| 182 | `Zotero/解_2023_文化元宇宙语境下的数字藏品运作机理与善治机制研究.pdf` | 解_2023_文化元宇宙语境下的数字藏品运作机理与善治机制研究.pdf | 文化元宇宙语境下的数字藏品运作机理与善治机制研究 | 解学芳\|祝新乐\|臧志彭 | 是 |
| 183 | `Zotero/解学芳 and 盖小飞_2017_技术创新、制度创新协同与文化产业发展综述与研判.pdf` | 解学芳 and 盖小飞_2017_技术创新、制度创新协同与文化产业发展综述与研判.pdf | 技术创新、制度创新协同与文化产业发展综述与研判 | 解学芳\|盖小飞 | 是 |
| 184 | `Zotero/解学芳_2015_文化创意产业科技创新环境优化研究——基于上海全球科技创新中心建设的思考.pdf` | 解学芳_2015_文化创意产业科技创新环境优化研究——基于上海全球科技创新中心建设的思考.pdf | 文化创意产业科技创新环境优化研究——基于上海全球科技创新中心建设的思考 | 解学芳 | 是 |
| 185 | `Zotero/言ルル_アンデパンダン展_20260529_4_f.pdf` | 言ルル_アンデパンダン展_20260529_4_f.pdf | 言ルル_アンデパンダン展_20260529_4_f |  | 是 |
| 186 | `Zotero/许_2022_时空破壁与场域重构面向文化数字化领域的元宇宙.pdf` | 许_2022_时空破壁与场域重构面向文化数字化领域的元宇宙.pdf | 时空破壁与场域重构面向文化数字化领域的元宇宙 | 许晶 | 是 |
| 187 | `Zotero/许峰_2025_城市数字化转型中的文化数字化发展研究——以政策演进为视角.pdf` | 许峰_2025_城市数字化转型中的文化数字化发展研究——以政策演进为视角.pdf | 城市数字化转型中的文化数字化发展研究——以政策演进为视角 | 许峰 | 是 |
| 188 | `Zotero/谢_2020_大数据成为现实生产要素的企业实现机制：产品创新视角.pdf` | 谢_2020_大数据成为现实生产要素的企业实现机制：产品创新视角.pdf | 大数据成为现实生产要素的企业实现机制：产品创新视角 | 谢康\|夏正豪\|肖静华 | 是 |
| 189 | `Zotero/谢永俊_2017_基于微博数据的北京市热点区域意象感知.pdf` | 谢永俊_2017_基于微博数据的北京市热点区域意象感知.pdf | 基于微博数据的北京市热点区域意象感知 | 谢永俊 | 是 |
| 190 | `Zotero/谭_2019_语义网络发展历程与现状研究.pdf` | 谭_2019_语义网络发展历程与现状研究.pdf | 语义网络发展历程与现状研究 | 谭荧\|张进\|夏立新 | 是 |
| 191 | `Zotero/贺_2023_华立科技游艺设备行业龙头  卡牌业务驱动成长.pdf` | 贺_2023_华立科技游艺设备行业龙头  卡牌业务驱动成长.pdf | 华立科技游艺设备行业龙头  卡牌业务驱动成长 | 贺天瑞 | 是 |
| 192 | `Zotero/赵_2014_信息社会的城市空间策略——智慧城市热潮的冷思考.pdf` | 赵_2014_信息社会的城市空间策略——智慧城市热潮的冷思考.pdf | 信息社会的城市空间策略——智慧城市热潮的冷思考 | 赵渺希\|王世福\|李璐颖 | 是 |
| 193 | `Zotero/赵_2015_互联网媒介中的城市意象图景——以广东21个城市为例.pdf` | 赵_2015_互联网媒介中的城市意象图景——以广东21个城市为例.pdf | 互联网媒介中的城市意象图景——以广东21个城市为例 | 赵渺希\|徐高峰\|李榕榕 | 是 |
| 194 | `Zotero/赵薇_2024_如何在“建造”中思辨——数字人文论争反思_1.pdf` | 赵薇_2024_如何在“建造”中思辨——数字人文论争反思_1.pdf | 如何在“建造”中思辨——数字人文论争反思_1 | 赵薇 | 是 |
| 195 | `Zotero/连 and 邓_2020_媒介、时间与现代性的“谎言”社会加速理论的传播批判研究.pdf` | 连 and 邓_2020_媒介、时间与现代性的“谎言”社会加速理论的传播批判研究.pdf | 媒介、时间与现代性的“谎言”社会加速理论的传播批判研究 | 连水兴\|邓丹 | 是 |
| 196 | `Zotero/邵 and 廖_2009_横店中国影视文化产业集群发展的一个样本——基于共享性资源观理论的案例分析.pdf` | 邵 and 廖_2009_横店中国影视文化产业集群发展的一个样本——基于共享性资源观理论的案例分析.pdf | 横店中国影视文化产业集群发展的一个样本——基于共享性资源观理论的案例分析 | 邵培仁\|廖卫民 | 是 |
| 197 | `Zotero/邹_2021_大数据环境下智慧城市信息安全保障体系构建研究.pdf` | 邹_2021_大数据环境下智慧城市信息安全保障体系构建研究.pdf | 大数据环境下智慧城市信息安全保障体系构建研究 | 邹凯\|郭一航\|向尚\|万震 | 是 |
| 198 | `Zotero/郭_2019_文化产业集聚、空间溢出与区域创新能力.pdf` | 郭_2019_文化产业集聚、空间溢出与区域创新能力.pdf | 文化产业集聚、空间溢出与区域创新能力 | 郭新茹\|顾江\|陈天宇 | 是 |
| 199 | `Zotero/郭_2019_过渡仪式视域下中国本命年文化习俗探究.pdf` | 郭_2019_过渡仪式视域下中国本命年文化习俗探究.pdf | 过渡仪式视域下中国本命年文化习俗探究 | 郭新茹\|王娜\|陈天宇 | 是 |
| 200 | `Zotero/郭_2021_文旅融合背景下我国文化产业园区高质量发展路径研究——以江苏为例.pdf` | 郭_2021_文旅融合背景下我国文化产业园区高质量发展路径研究——以江苏为例.pdf | 文旅融合背景下我国文化产业园区高质量发展路径研究——以江苏为例 | 郭新茹\|沈佳\|陈天宇 | 是 |
| 201 | `Zotero/郭_2022_从技术中心主义到人本主义：智慧城市研究进展与展望.pdf` | 郭_2022_从技术中心主义到人本主义：智慧城市研究进展与展望.pdf | 从技术中心主义到人本主义：智慧城市研究进展与展望 | 郭杰\|王珺\|姜璐\|张虹鸥\|黄耿志 | 是 |
| 202 | `Zotero/郭新茹 and 陈天宇_2020_长三角文化市场区域合作与一体化路径研究.pdf` | 郭新茹 and 陈天宇_2020_长三角文化市场区域合作与一体化路径研究.pdf | 长三角文化市场区域合作与一体化路径研究 | 郭新茹\|陈天宇 | 是 |
| 203 | `Zotero/郭新茹 and 陈天宇_2021_地理集聚能否提升我国文化产业发展质量——基于空间杜宾模型和门槛模型的检验.pdf` | 郭新茹 and 陈天宇_2021_地理集聚能否提升我国文化产业发展质量——基于空间杜宾模型和门槛模型的检验.pdf | 地理集聚能否提升我国文化产业发展质量——基于空间杜宾模型和门槛模型的检验 | 郭新茹\|陈天宇 | 是 |
| 204 | `Zotero/郭新茹_2021_场景视域下大运河非遗生活性保护的策略研究.pdf` | 郭新茹_2021_场景视域下大运河非遗生活性保护的策略研究.pdf | 场景视域下大运河非遗生活性保护的策略研究 | 郭新茹 | 是 |
| 205 | `Zotero/金_2016_“同时代性”文学批评的“起跑线”.pdf` | 金_2016_“同时代性”文学批评的“起跑线”.pdf | “同时代性”文学批评的“起跑线” | 金理 | 是 |
| 206 | `Zotero/闫宏秀 and 宋胜男_2024_双重脆弱性与适度信任：从ChatGPT到Sora_1.pdf` | 闫宏秀 and 宋胜男_2024_双重脆弱性与适度信任：从ChatGPT到Sora_1.pdf | 双重脆弱性与适度信任：从ChatGPT到Sora_1 | 闫宏秀\|宋胜男 | 是 |
| 207 | `Zotero/陆_2019_《2018上海智慧城市发展水平评估报告》解析.pdf` | 陆_2019_《2018上海智慧城市发展水平评估报告》解析.pdf | 《2018上海智慧城市发展水平评估报告》解析 | 陆森\|刘岩\|辛竹 | 是 |
| 208 | `Zotero/陈 and 黄_2018_协同治理我国城市社区公共文化服务供给机制创新的个案研究.pdf` | 陈 and 黄_2018_协同治理我国城市社区公共文化服务供给机制创新的个案研究.pdf | 协同治理我国城市社区公共文化服务供给机制创新的个案研究 | 陈世香\|黄冬季 | 是 |
| 209 | `Zotero/陈_2015_基本公共文化服务概念演变与协同.pdf` | 陈_2015_基本公共文化服务概念演变与协同.pdf | 基本公共文化服务概念演变与协同 | 陈昊琳 | 是 |
| 210 | `Zotero/陈_2020_从赋能到使能——数字化环境下的企业运营管理.pdf` | 陈_2020_从赋能到使能——数字化环境下的企业运营管理.pdf | 从赋能到使能——数字化环境下的企业运营管理 | 陈剑\|黄朔\|刘运辉 | 是 |
| 211 | `Zotero/陈_2020_大数据环境下的决策范式转变与使能创新.pdf` | 陈_2020_大数据环境下的决策范式转变与使能创新.pdf | 大数据环境下的决策范式转变与使能创新 | 陈国青\|曾大军\|卫强\|张明月\|郭迅华 | 是 |
| 212 | `Zotero/陈_2021_《美国数字公共图书馆2019—2022年战略规划》分析及启示.pdf` | 陈_2021_《美国数字公共图书馆2019—2022年战略规划》分析及启示.pdf | 《美国数字公共图书馆2019—2022年战略规划》分析及启示 | 陈汝南\|刘晓莹\|李松芮 | 是 |
| 213 | `Zotero/陈_2021_创新型城市管理与决策研究前沿——第234期双清论坛学术综述.pdf` | 陈_2021_创新型城市管理与决策研究前沿——第234期双清论坛学术综述.pdf | 创新型城市管理与决策研究前沿——第234期双清论坛学术综述 | 陈晓红\|聂建国\|吴志强\|杨列勋\|刘作仪\|任之光\|陈彬 | 是 |
| 214 | `Zotero/陈_2021_基于动态语义网络分析的主题演化路径识别研究.pdf` | 陈_2021_基于动态语义网络分析的主题演化路径识别研究.pdf | 基于动态语义网络分析的主题演化路径识别研究 | 陈翔\|黄璐\|倪兴兴\|刘家润\|曹晓丽\|王长天 | 是 |
| 215 | `Zotero/陈_2022_数字赋能文化产业的发展趋势与策略选择.pdf` | 陈_2022_数字赋能文化产业的发展趋势与策略选择.pdf | 数字赋能文化产业的发展趋势与策略选择 | 陈知然\|庞亚君\|周雪\|刘伟杰 | 是 |
| 216 | `Zotero/陈勇 and 李想_2026_导航与探索：AI时代网络文学中的人机协同创作.pdf` | 陈勇 and 李想_2026_导航与探索：AI时代网络文学中的人机协同创作.pdf | 导航与探索：AI时代网络文学中的人机协同创作 | 陈勇\|李想 | 是 |
| 217 | `Zotero/陶_2023_展高机甲+卡牌  创领变形玩具新风潮.pdf` | 陶_2023_展高机甲+卡牌  创领变形玩具新风潮.pdf | 展高机甲+卡牌  创领变形玩具新风潮 | 陶鑫 | 是 |
| 218 | `Zotero/韦_2021_“沉浸式”的消费与革新当代戏剧观演关系批判.pdf` | 韦_2021_“沉浸式”的消费与革新当代戏剧观演关系批判.pdf | “沉浸式”的消费与革新当代戏剧观演关系批判 | 韦哲宇 | 是 |
| 219 | `Zotero/韦_2022_智慧城市建设对企业研发操纵的影响研究.pdf` | 韦_2022_智慧城市建设对企业研发操纵的影响研究.pdf | 智慧城市建设对企业研发操纵的影响研究 | 韦琳\|马梦茹\|金宇 | 是 |
| 220 | `Zotero/马睿 - 2025 - 艺术史视域下的数字技术及其美学伦理.pdf` | 马睿 - 2025 - 艺术史视域下的数字技术及其美学伦理.pdf | 马睿 - 2025 - 艺术史视域下的数字技术及其美学伦理 |  | 是 |
| 221 | `Zotero/高_2023_数字文化新范式与人文创新研究.pdf` | 高_2023_数字文化新范式与人文创新研究.pdf | 数字文化新范式与人文创新研究 | 高峰\|徐迎庆\|刘梦庭\|吴昱苇 | 是 |
| 222 | `Zotero/高峰 - 2026 - 概率建模视角下的AIGC艺术生成——人机共创的训练机制与创作范式重构.pdf` | 高峰 - 2026 - 概率建模视角下的AIGC艺术生成——人机共创的训练机制与创作范式重构.pdf | 高峰 - 2026 - 概率建模视角下的AIGC艺术生成——人机共创的训练机制与创作范式重构 |  | 是 |
| 223 | `Zotero/魏_2020_城市文化空间塑造的国际经验与启示——以伦敦、纽约、巴黎、东京为例.pdf` | 魏_2020_城市文化空间塑造的国际经验与启示——以伦敦、纽约、巴黎、东京为例.pdf | 城市文化空间塑造的国际经验与启示——以伦敦、纽约、巴黎、东京为例 | 魏伟\|刘畅\|张帅权\|王兵 | 是 |
| 224 | `Zotero/鹿_2022_利益相关者视角下社会责任履责对企业财务绩效的影响研究——基于文化娱乐产业的实证分析.pdf` | 鹿_2022_利益相关者视角下社会责任履责对企业财务绩效的影响研究——基于文化娱乐产业的实证分析.pdf | 利益相关者视角下社会责任履责对企业财务绩效的影响研究——基于文化娱乐产业的实证分析 | 鹿立阳\|张静茹\|崔运周 | 是 |
| 225 | `Zotero/黄_2016_“恶之花”清末民初上海夜花园与女学生的“黑幕叙事”.pdf` | 黄_2016_“恶之花”清末民初上海夜花园与女学生的“黑幕叙事”.pdf | “恶之花”清末民初上海夜花园与女学生的“黑幕叙事” | 黄湘金 | 是 |
| 226 | `Zotero/黄_2022_乡村振兴下的四川乡村文旅融合：困境、路径及模式.pdf` | 黄_2022_乡村振兴下的四川乡村文旅融合：困境、路径及模式.pdf | 乡村振兴下的四川乡村文旅融合：困境、路径及模式 | 黄萍\|黎玲\|胡珑川 | 是 |
| 227 | `Zotero/黄_2023_文化数字化的多维观察与前瞻(笔谈).pdf` | 黄_2023_文化数字化的多维观察与前瞻(笔谈).pdf | 文化数字化的多维观察与前瞻(笔谈) | 黄永林\|宋俊华\|张士闪\|肖远平 | 是 |
| 228 | `Zotero/黄_2024_数字赋能文旅深度融合的理论逻辑与研究框架.pdf` | 黄_2024_数字赋能文旅深度融合的理论逻辑与研究框架.pdf | 数字赋能文旅深度融合的理论逻辑与研究框架 | 黄震方\|张子昂\|李涛\|侯国林\|靳诚\|张郴 | 是 |
| 229 | `Zotero/龙_2020_中国智慧城市发展现状及未来发展趋势研究.pdf` | 龙_2020_中国智慧城市发展现状及未来发展趋势研究.pdf | 中国智慧城市发展现状及未来发展趋势研究 | 龙瀛\|张雨洋\|张恩嘉\|陈议威 | 是 |

## OneDrive/（80 行）

| # | 原路径 | 文件名 | 标题 | 作者 | 父目录在盘 |
|---:|---|---|---|---|---|
| 1 | `OneDrive/Albert Borgmann/Technology and the Character of Contemporary Life_ (8250)/Technology and the Character of Conte` | Technology and the Character of Contemporary Life_ - Albert Borgmann.pdf#9f99dbfe-3135-459f-8bca-28eeede448fe | Technology and the Character of Contemporary Life: A Philosophical Inquiry | Albert Borgmann | 是 |
| 2 | `OneDrive/Alexander C Karp/The Technological Republic_ Hard Power, Soft Belief, and the Future of the West (9842)/The Tec` | The Technological Republic_ Hard Power, So - Alexander C Karp.epub | The Technological Republic: Hard Power, Soft Belief, and the Future of the West | Alexander C Karp\|Nicholas W. Zamiska | 否 |
| 3 | `OneDrive/Alexander R. Galloway/Uncomputable_ Play and Politics in the Long Digita (8644)/Uncomputable_ Play and Politics` | Uncomputable_ Play and Politics in the Lon - Alexander R. Galloway.epub | Uncomputable: Play and Politics in the Long Digital Age | Alexander R. Galloway | 是 |
| 4 | `OneDrive/Arthur Asa Berger/What objects mean _ an introduction to material cu (10244)/What objects mean _ an introductio` | What objects mean _ an introduction to material cu - Arthur Asa Berger.epub#94ed7ba3-e2fc-4254-bed9-f5e842c70c08 | What objects mean : an introduction to material culture | Arthur Asa Berger | 是 |
| 5 | `OneDrive/Benjamin H. Bratton/真实政治的复仇 (10022)/真实政治的复仇 - Benjamin H. Bratton.epub.epub` | 真实政治的复仇 - Benjamin H. Bratton.epub.epub | 真实政治的复仇 | Benjamin H. Bratton | 是 |
| 6 | `OneDrive/Brian Hurt/想看 在看 看过 (6850)/想看 在看 看过 - Brian Hurt.epub.epub` | 想看 在看 看过 - Brian Hurt.epub.epub | 想看 在看 看过 | Brian Hurt | 是 |
| 7 | `OneDrive/Charlotte Perkins Gilman/黄色墙纸：吉尔曼女性小说选 (9714)/黄色墙纸：吉尔曼女性小说选 - Charlotte Perkins Gilman.epub.epub` | 黄色墙纸：吉尔曼女性小说选 - Charlotte Perkins Gilman.epub.epub | 黄色墙纸：吉尔曼女性小说选 | Charlotte Perkins Gilman | 是 |
| 8 | `OneDrive/Dan Voinea/永恒和一日：安哲罗普洛斯访谈录 (9797)/永恒和一日：安哲罗普洛斯访谈录 - Dan Voinea.epub.epub` | 永恒和一日：安哲罗普洛斯访谈录 - Dan Voinea.epub.epub | 永恒和一日：安哲罗普洛斯访谈录 | Dan Voinea | 是 |
| 9 | `OneDrive/Dava Sobel/经度：一个孤独的天才解决所处时代最大难题的真实故事 (9863)/经度：一个孤独的天才解决所处 - Dava Sobel.epub` | 经度：一个孤独的天才解决所处 - Dava Sobel.epub | 经度：一个孤独的天才解决所处时代最大难题的真实故事 | Dava Sobel | 是 |
| 10 | `OneDrive/David Richo/How to Be an Adult in Relationships (4957)/How to Be an Adult in Relationships - David Richo.epub.e` | How to Be an Adult in Relationships - David Richo.epub.epub | How to Be an Adult in Relationships | David Richo | 是 |
| 11 | `OneDrive/Derek Attridge/The Craft of Poetry_ Dialogues on Minimal Interpre (8174)/The Craft of Poetry_ Dialogues on Mini` | The Craft of Poetry_ Dialogues on Minimal Interpre - Derek Attridge.epub.epub | The Craft of Poetry: Dialogues on Minimal Interpretation | Henry Staten\|Derek Attridge | 是 |
| 12 | `OneDrive/Derk Peremboom/自由意志、能动性与生命的意义 (10303)/自由意志、能动性与生命的意义 - Derk Peremboom.epub` | 自由意志、能动性与生命的意义 - Derk Peremboom.epub | 自由意志、能动性与生命的意义 | Derk Peremboom | 否 |
| 13 | `OneDrive/Domique Cardon/Culture Numérique (9986)/Culture Numérique - Domique Cardon.epub` | Culture Numérique - Domique Cardon.epub | Culture Numérique | Domique Cardon | 否 |
| 14 | `OneDrive/Dong Hao Ji/gemuDe riarizumunoDan Sheng  Dong Wu Hua suruposutomodan2 (Jiang Tan She Xian Dai Xin Shu ) (9938)/` | gemuDe riarizumunoDan Sheng  Dong Wu Hua s - Dong Hao Ji.epub | ゲーム的リアリズムの誕生 動物化するポストモダン２ (講談社現代新書) | Dong Hao Ji | 否 |
| 15 | `OneDrive/Durham Peters John/Speaking into the Air_ A History of the Idea of Co (1634)/Speaking into the Air_ A History o` | Speaking into the Air_ A History of the Idea of Co - Durham Peters John.epub | Speaking into the Air: A History of the Idea of Communication | Durham Peters John | 否 |
| 16 | `OneDrive/Dustin A. Abnet/The American Robot (9532)/The American Robot - Dustin A. Abnet.epub.epub` | The American Robot - Dustin A. Abnet.epub.epub | The American Robot | Dustin A. Abnet | 是 |
| 17 | `OneDrive/Ellen Dissanayake/Homo aestheticus where art comes from and why (3412)/Homo aestheticus where art comes from an` | Homo aestheticus where art comes from and why - Ellen Dissanayake.epub.epub | Homo aestheticus where art comes from and why | Ellen Dissanayake | 是 |
| 18 | `OneDrive/Feenberg Andrew/Alternative Modernity The Technical Turn in Philos (742)/Alternative Modernity The Technical Tu` | Alternative Modernity The Technical Turn in Philos - Feenberg Andrew.epub | Alternative Modernity The Technical Turn in Philosophy and Social Theory | Feenberg Andrew | 否 |
| 19 | `OneDrive/Francesca Cernia Slovin/Obsessed by Art_ Aby Warburg_ His Life and His Leg (6854)/Obsessed by Art_ Aby Warburg_` | Obsessed by Art_ Aby Warburg_ His Life and His Leg - Francesca Cernia Slovin.epub.epub | Obsessed by Art: Aby Warburg: His Life and His Legacy | Francesca Cernia Slovin | 是 |
| 20 | `OneDrive/Geoffrey Engelstein/Achievement Relocked_ Loss aversion and game desig (10187)/Achievement Relocked_ Loss avers` | Achievement Relocked_ Loss aversion and game desig - Geoffrey Engelstein.epub.epub | Achievement Relocked: Loss aversion and game design | Geoffrey Engelstein | 是 |
| 21 | `OneDrive/Geoffrey Engelstein/Achievement Relocked_ Loss aversion and game desig (10187)/Achievement Relocked_ Loss avers` | Achievement Relocked_ Loss aversion and game desig - Geoffrey Engelstein.epub.epub#ac93cf01-31be-428c-a62c-65b7c148d563 | Achievement Relocked: Loss aversion and game design | Geoffrey Engelstein | 是 |
| 22 | `OneDrive/Geoffrey Engelstein/Building Blocks of Tabletop Game Design (10188)/Building Blocks of Tabletop Game Design - G` | Building Blocks of Tabletop Game Design - Geoffrey Engelstein.pdf#494c3bac-85f2-4095-a949-42bb07552304 | Building Blocks of Tabletop Game Design | Geoffrey Engelstein | 是 |
| 23 | `OneDrive/Gerd Breitenberg/伊加利亚的女儿们 (10261)/伊加利亚的女儿们 - Gerd Breitenberg.epub.epub` | 伊加利亚的女儿们 - Gerd Breitenberg.epub.epub | 伊加利亚的女儿们 | Gerd Breitenberg | 是 |
| 24 | `OneDrive/Herbert W. Franke/Zentrum der Milchstraße (7546)/Zentrum der Milchstraße - Herbert W. Franke.epub.epub` | Zentrum der Milchstraße - Herbert W. Franke.epub.epub | Zentrum der Milchstraße | Herbert W. Franke | 是 |
| 25 | `OneDrive/Hong Fincher Leta/Betraying Big Brother The Feminist Awakening in Ch (1524)/Betraying Big Brother The Femin - H` | Betraying Big Brother The Femin - Hong Fincher Leta.epub | Betraying Big Brother The Feminist Awakening in China-Verso (2018) | Hong Fincher Leta | 是 |
| 26 | `OneDrive/Immanuel Kant/Allgemeine Naturgeschichte und Theorie des Himmels (6477)/Allgemeine Naturgeschichte und - Immanu` | Allgemeine Naturgeschichte und - Immanuel Kant.epub | Allgemeine Naturgeschichte und Theorie des Himmels | Immanuel Kant | 是 |
| 27 | `OneDrive/Isaac Asimov/阿西莫夫科幻短篇全集：2：双百人（全二册） (9736)/阿西莫夫科幻短篇全集：2：双 - Isaac Asimov.epub` | 阿西莫夫科幻短篇全集：2：双 - Isaac Asimov.epub | 阿西莫夫科幻短篇全集：2：双百人（全二册） | Isaac Asimov | 是 |
| 28 | `OneDrive/Janice A. Radway/阅读浪漫小说：女性，父权制和通俗文学 (9383)/阅读浪漫小说：女性，父权制和 - Janice A. Radway.epub` | 阅读浪漫小说：女性，父权制和 - Janice A. Radway.epub | 阅读浪漫小说：女性，父权制和通俗文学 | Janice A. Radway | 是 |
| 29 | `OneDrive/Jenny L. Davis/How Artifacts Afford (8600)/How Artifacts Afford - Jenny L. Davis.epub.epub` | How Artifacts Afford - Jenny L. Davis.epub.epub | How Artifacts Afford | Jenny L. Davis | 是 |
| 30 | `OneDrive/Julia Voss/Why Art Criticism_ A Reader (8957)/Why Art Criticism_ A Reader - Beate Söntgen.epub` | Why Art Criticism_ A Reader - Beate Söntgen.epub | Why Art Criticism? A Reader | Beate Söntgen\|Julia Voss | 是 |
| 31 | `OneDrive/Lanier Jaron/Dawn of the New Everything Encounters with Reality (651)/Dawn of the New Everything Encounters wit` | Dawn of the New Everything Encounters with Reality - Lanier Jaron.epub | Dawn of the New Everything Encounters with Reality and Virtual Reality-Henry Holt and Co | Lanier Jaron | 否 |
| 32 | `OneDrive/Laura Bates/Men Who Hate Women, Girl Up, Everyday Sexism 3 Boo (9455)/Men Who Hate Women, Girl Up, Everyday Sex` | Men Who Hate Women, Girl Up, Everyday Sexism 3 Boo - Laura Bates.epub.epub | Men Who Hate Women, Girl Up, Everyday Sexism 3 Books Collection Set | Laura Bates | 是 |
| 33 | `OneDrive/Louisa Yousfi/埃德加·莫林：思考的冒险 =  Edgar Morin. L'aventure d'une pensée (10125)/埃德加·莫林：思考的冒险 =  Edg - Louisa Yousfi.` | 埃德加·莫林：思考的冒险 =  Edg - Louisa Yousfi.epub | 埃德加·莫林：思考的冒险 =  Edgar Morin. L'aventure d'une pensée | Louisa Yousfi | 否 |
| 34 | `OneDrive/Marjorie B. Garber/Academic Instincts (9908)/Academic Instincts - Marjorie B. Garber.epub#ff16d2c4-1859-4d89-84` | Academic Instincts - Marjorie B. Garber.epub#ff16d2c4-1859-4d89-8444-ed529dcdd245 | Academic Instincts | Marjorie B. Garber | 是 |
| 35 | `OneDrive/Marres Noortje/Digital Sociology The Reinvention of Social Resear (1346)/Digital Sociology The Reinvention of S` | Digital Sociology The Reinvention of Social Resear - Marres Noortje.epub | Digital Sociology The Reinvention of Social Research (2017, Polity) | Marres Noortje | 否 |
| 36 | `OneDrive/Mary L. Gray/Ghost Work_ How to Stop Silicon Valley From Buildi (1487)/Ghost Work_ How to Stop Silicon - Mary L` | Ghost Work_ How to Stop Silicon - Mary L. Gray.epub | Ghost Work: How to Stop Silicon Valley From Building a New Global Underclass | Mary L. Gray\|Siddharth Suri | 是 |
| 37 | `OneDrive/Maximilian Kasy/预测的方法 = The Means of Prediction_ How AI Really Wor (10147)/预测的方法 = The Means of Prediction_ How` | 预测的方法 = The Means of Prediction_ How AI Really Wor - Maximilian Kasy.epub.epub | 预测的方法 = The Means of Prediction: How AI Really Works (and Who Benefits) | Maximilian Kasy | 是 |
| 38 | `OneDrive/Michael Gibbons/知识生产的新模式_当代社会科学与研究的动力学 (科学技术政策译丛) (6858)/Zhi Shi Sheng Chan De Xin Mo Sh - Michael Gibbons.epub` | Zhi Shi Sheng Chan De Xin Mo Sh - Michael Gibbons.epub | 知识生产的新模式:当代社会科学与研究的动力学 (科学技术政策译丛) | Michael Gibbons | 否 |
| 39 | `OneDrive/Nina Bandelj/过度投资 Overinvested_ The Emotion Al Economy of Modern Parenting (10061)/过度投资 Overinvested_ The Emoti` | 过度投资 Overinvested_ The Emotion Al - Nina Bandelj.epub | 过度投资 Overinvested: The Emotion Al Economy of Modern Parenting | Nina Bandelj | 否 |
| 40 | `OneDrive/Pierre Nora/记忆之场 法国国民意识的文化社会史 (1474)/Ji Yi Zhi Chang  Fa Guo Guo Min - Pierre Nora.epub` | Ji Yi Zhi Chang  Fa Guo Guo Min - Pierre Nora.epub | 记忆之场 法国国民意识的文化社会史 | Pierre Nora | 是 |
| 41 | `OneDrive/Prelinger Megan/Inside the Machine Art and Invention in the Electr (3132)/Inside the Machine Art and Inve - Pre` | Inside the Machine Art and Inve - Prelinger Megan.epub | Inside the Machine Art and Invention in the Electronic Age | Megan Prelinger | 否 |
| 42 | `OneDrive/Prelinger Megan/Inside the Machine Art and Invention in the Electr (3132)/Inside the Machine Art and Invention ` | Inside the Machine Art and Invention in the Electr - Prelinger Megan.epub | Inside the Machine Art and Invention in the Electronic Age | Megan Prelinger | 否 |
| 43 | `OneDrive/Rebecca Gibson/Desire in the Age of Robots and AI (10196)/Desire in the Age of Robots and AI - Rebecca Gibson.e` | Desire in the Age of Robots and AI - Rebecca Gibson.epub.epub | Desire in the Age of Robots and AI | Rebecca Gibson | 是 |
| 44 | `OneDrive/Rebecca Gibson/Desire in the Age of Robots and AI (10196)/Desire in the Age of Robots and AI - Rebecca Gibson.e` | Desire in the Age of Robots and AI - Rebecca Gibson.epub.epub#cdb89226-69c8-491c-a499-f81c4b452a79 | Desire in the Age of Robots and AI | Rebecca Gibson | 是 |
| 45 | `OneDrive/Richard Coyne/AI and Language in the Urban Context; Conversation (10192)/AI and Language in the Urban Context; ` | AI and Language in the Urban Context; Conversation - Richard Coyne.epub.epub | AI and Language in the Urban Context; Conversational Artificial Intelligence in Cities | Richard Coyne | 是 |
| 46 | `OneDrive/Richard Shusterman/Analytic Aesthetics (2998)/Analytic Aesthetics - Shusterman Richard - 2026-08-24T024513.4372` | Analytic Aesthetics - Shusterman Richard - 2026-08-24T024513.437297Z.pdf | Analytic Aesthetics | Richard Shusterman | 是 |
| 47 | `OneDrive/Robert McDermott/American Philosophy and Rudolf Steiner (30)/American Philosophy and Rudolf - Robert McDermott.` | American Philosophy and Rudolf - Robert McDermott.epub | American Philosophy and Rudolf Steiner | Robert McDermott | 是 |
| 48 | `OneDrive/Rudolf Steiner/The Karma of Untruthfulness_ v. 1 (18)/The Karma of Untruthfulness_ v. - Rudolf Steiner.epub` | The Karma of Untruthfulness_ v. - Rudolf Steiner.epub | The Karma of Untruthfulness: v. 1 | Rudolf Steiner | 是 |
| 49 | `OneDrive/Saikat Majumdar/The Amateur_ Self-making and the humanities in the postcolony (9906)/The Amateur_ Self-making a` | The Amateur_ Self-making and the humanitie - Saikat Majumdar.epub | The Amateur: Self-making and the humanities in the postcolony | Saikat Majumdar | 否 |
| 50 | `OneDrive/Sherry Turkle/重拾交谈 (8393)/重拾交谈 - [美]雪莉·特克尔.epub` | 重拾交谈 - [美]雪莉·特克尔.epub | 重拾交谈 | Sherry Turkle | 是 |
| 51 | `OneDrive/Slavoj Žižek/The Idea of Communism (1334)/The Idea of Communism - Slavoj Žižek - 2026-08-24T023630.066653Z.pdf` | The Idea of Communism - Slavoj Žižek - 2026-08-24T023630.066653Z.pdf | The Idea of Communism | Slavoj Žižek | 是 |
| 52 | `OneDrive/Stanisław Lem/机器人大师 (6973)/机器人大师 - Stanisław Lem - 2026-08-23T231841.730082Z.pdf` | 机器人大师 - Stanisław Lem - 2026-08-23T231841.730082Z.pdf | 机器人大师 | Stanisław Lem | 是 |
| 53 | `OneDrive/Stanley Coren/狗故事_人类历史上狗的爪印 (9335)/狗故事_人类历史上狗的爪印 - Stanley Coren - 2026-08-24T013532.195238Z.pdf` | 狗故事_人类历史上狗的爪印 - Stanley Coren - 2026-08-24T013532.195238Z.pdf | 狗故事_人类历史上狗的爪印 | Stanley Coren | 是 |
| 54 | `OneDrive/Stephen Hawking/时空的未来 (9292)/时空的未来 - Stephen Hawking - 2026-08-24T012342.108074Z.pdf` | 时空的未来 - Stephen Hawking - 2026-08-24T012342.108074Z.pdf | 时空的未来 | Stephen Hawking | 是 |
| 55 | `OneDrive/Stephen Hawking/霍金讲演录_黑洞、婴儿宇宙及其他 (9288)/霍金讲演录_黑洞、婴儿宇宙及其他 - Stephen Hawking - 2026-08-24T005531.055235Z.pdf` | 霍金讲演录_黑洞、婴儿宇宙及其他 - Stephen Hawking - 2026-08-24T005531.055235Z.pdf | 霍金讲演录_黑洞、婴儿宇宙及其他 | Stephen Hawking | 是 |
| 56 | `OneDrive/Ted Chiang/你一生的故事 (2296)/你一生的故事 - Ted Chiang - 2026-08-23T151924.605415Z.pdf` | 你一生的故事 - Ted Chiang - 2026-08-23T151924.605415Z.pdf | 你一生的故事 | Ted Chiang | 是 |
| 57 | `OneDrive/Theodor W. Adorno/本真性的黑话：评德意志意识形态 (7830)/本真性的黑话：评德意志意识形态 (7830).epub` | 本真性的黑话：评德意志意识形态 (7830).epub | 本真性的黑话：评德意志意识形态 | Theodor W. Adorno | 是 |
| 58 | `OneDrive/Theodor W. Adorno/本真性的黑话：评德意志意识形态 (7830)/本真性的黑话：评德意志意识形态 - Theodor W. Adorno.epub#87f2d007-ea42-459c-8fcc-f53a1` | 本真性的黑话：评德意志意识形态 - Theodor W. Adorno.epub#87f2d007-ea42-459c-8fcc-f53a1808e525 | 本真性的黑话：评德意志意识形态 | Theodor W. Adorno | 是 |
| 59 | `OneDrive/Thomas Rid/Rise of the Machines_ A Cybernetic History (193)/Rise of the Machines_ A Cyberne - Thomas Rid.epub` | Rise of the Machines_ A Cyberne - Thomas Rid.epub | Rise of the Machines: A Cybernetic History | Thomas Rid | 是 |
| 60 | `OneDrive/Umberto Eco/美的历史·彩图版 (2349)/美的历史·彩图版 - Umberto Eco - 2026-08-23T154007.594292Z.pdf` | 美的历史·彩图版 - Umberto Eco - 2026-08-23T154007.594292Z.pdf | 美的历史·彩图版 | Umberto Eco | 是 |
| 61 | `OneDrive/Uwe Schutte/Kraftwerk; Future Music from Germany (10064)/Kraftwerk; Future Music from Germany - Uwe Schutte.epu` | Kraftwerk; Future Music from Germany - Uwe Schutte.epub | Kraftwerk; Future Music from Germany | Uwe Schütte | 否 |
| 62 | `OneDrive/Wolfgang Schivelbusch/The Railway Journey_ The Industrialization of Time (1255)/The Railway Journey_ The Indust` | The Railway Journey_ The Indust - Wolfgang Schivelbusch.epub | The Railway Journey: The Industrialization of Time and Space in the Nineteenth Century | Wolfgang Schivelbusch | 是 |
| 63 | `OneDrive/【法】西蒙娜·德·波伏瓦/青春手记 (9718)/青春手记 - 【法】西蒙娜·德·波伏瓦.epub` | 青春手记 - 【法】西蒙娜·德·波伏瓦.epub | 青春手记 | Simone de Beauvoir | 否 |
| 64 | `OneDrive/东浩纪/ゲンロン0 观光客的哲学 epub (7179)/ゲンロン0 观光客的哲学 epub - 东浩纪.epub` | ゲンロン0 观光客的哲学 epub - 东浩纪.epub | ゲンロン0 观光客的哲学 epub | 东浩纪 | 否 |
| 65 | `OneDrive/伊藤润二/伊藤润二的漫画术：恐怖诞生的地方 (9793)/伊藤润二的漫画术：恐怖诞生的 - 伊藤润二.epub` | 伊藤润二的漫画术：恐怖诞生的 - 伊藤润二.epub | 伊藤润二的漫画术：恐怖诞生的地方 | 伊藤润二 | 是 |
| 66 | `OneDrive/劳拉·贝茨/隐秘的角落：无孔不入的厌女症 (9795)/隐秘的角落：无孔不入的厌女症 - 劳拉·贝茨.epub` | 隐秘的角落：无孔不入的厌女症 - 劳拉·贝茨.epub | 隐秘的角落：无孔不入的厌女症 | Laura Bates | 否 |
| 67 | `OneDrive/史蒂芬·都伯纳/魔鬼经济学系列(套装共4册)-中信出版集团 (2016) (746)/魔鬼经济学系列(套装共4册)-中 - 史蒂芬·都伯纳.epub` | 魔鬼经济学系列(套装共4册)-中 - 史蒂芬·都伯纳.epub | 魔鬼经济学系列(套装共4册)-中信出版集团 (2016) | Stephen Dubner | 否 |
| 68 | `OneDrive/多多/词语磁场：多多五十年诗歌自选集（1972-2022） (9706)/词语磁场：多多五十年诗歌自选 - 多多.epub` | 词语磁场：多多五十年诗歌自选 - 多多.epub | 词语磁场：多多五十年诗歌自选集（1972-2022） | 多多 | 是 |
| 69 | `OneDrive/安·拉莫特/一只鸟接着一只鸟：关于写作与人生的建议 (8435)/一只鸟接着一只鸟：关于写作与 - 安·拉莫特.epub` | 一只鸟接着一只鸟：关于写作与 - 安·拉莫特.epub | 一只鸟接着一只鸟：关于写作与人生的建议 | Anne Lamott | 否 |
| 70 | `OneDrive/尤瓦尔·赫拉利/未来简史 (356)/Wei Lai Jian Shi - You Wa Er _He La Li.epub` | Wei Lai Jian Shi - You Wa Er _He La Li.epub | 未来简史 | Yuval Noah Harari | 否 |
| 71 | `OneDrive/成庆/150234_人生解忧：佛学入门四十讲_成庆 (9720)/150234_人生解忧：佛学入门四十讲_成庆 - 成庆.epub.epub` | 150234_人生解忧：佛学入门四十讲_成庆 - 成庆.epub.epub | 150234_人生解忧：佛学入门四十讲_成庆 | 成庆 | 是 |
| 72 | `OneDrive/未知/事发之室：白宫回忆录 (1409)/Shi Fa Zhi Shi _Bai Gong Hui Yi - Wei Zhi.epub` | Shi Fa Zhi Shi _Bai Gong Hui Yi - Wei Zhi.epub | 事发之室：白宫回忆录 | John Bolton | 否 |
| 73 | `OneDrive/未知/代数数论的原始导引 (2583)/代数数论的原始导引 - 未知 - 2026-08-23T153410.993971Z.pdf` | 代数数论的原始导引 - 未知 - 2026-08-23T153410.993971Z.pdf | 代数数论的原始导引 | 未知 | 是 |
| 74 | `OneDrive/未知/元代史学思想研 (5391)/元代史学思想研 - 未知 - 2026-08-23T164614.928809Z.pdf` | 元代史学思想研 - 未知 - 2026-08-23T164614.928809Z.pdf | 元代史学思想研 | 未知 | 是 |
| 75 | `OneDrive/朱立元/艺术美学辞典 (10353)/艺术美学辞典 - 朱立元 - 2026-08-24T023331.306284Z.pdf` | 艺术美学辞典 - 朱立元 - 2026-08-24T023331.306284Z.pdf | 艺术美学辞典 | 朱立元 | 是 |
| 76 | `OneDrive/理查德·奥斯曼/周四推理俱乐部：“魔鬼”的最后一眼 (9732)/周四推理俱乐部：“魔鬼”的最 - 理查德·奥斯曼.epub` | 周四推理俱乐部：“魔鬼”的最 - 理查德·奥斯曼.epub | 周四推理俱乐部：“魔鬼”的最后一眼 | Richard Osman | 否 |
| 77 | `OneDrive/瑞·达利欧/国家为什么会破产：大周期 (9753)/国家为什么会破产：大周期 - 瑞·达利欧.epub` | 国家为什么会破产：大周期 - 瑞·达利欧.epub | 国家为什么会破产：大周期 | Ray Dalio | 否 |
| 78 | `OneDrive/西尔维·库尔廷·德纳米/黑暗时代三女哲：施泰因、阿伦特、韦伊评传 (8497)/黑暗时代三女哲：施泰因、阿伦 - 西尔维·库尔廷·德纳米.epub` | 黑暗时代三女哲：施泰因、阿伦 - 西尔维·库尔廷·德纳米.epub | 黑暗时代三女哲：施泰因、阿伦特、韦伊评传 | Sylvie Courtine-Denamy | 否 |
| 79 | `OneDrive/许煜/Recursivity and Contingency (6556)/Recursivity and Contingency - Xu Yu.epub` | Recursivity and Contingency - Xu Yu.epub | Recursivity and Contingency | 许煜 | 是 |
| 80 | `OneDrive/雷沙德·卡普希钦斯基/十一个时区之旅：北国五十年 (9774)/十一个时区之旅：北国五十年 - 雷沙德·卡普希钦斯基.epub` | 十一个时区之旅：北国五十年 - 雷沙德·卡普希钦斯基.epub | 十一个时区之旅：北国五十年 | Ryszard Kapuściński | 否 |

## Series/（15 行）

| # | 原路径 | 文件名 | 标题 | 作者 | 父目录在盘 |
|---:|---|---|---|---|---|
| 1 | `Series/桐城派名家文集/桐城派名家文集 1 姚范集 方树东集 吴德旋集.pdf` | 桐城派名家文集 1 姚范集 方树东集 吴德旋集.pdf | 桐城派名家文集 1 姚范集 方树东集 吴德旋集 |  | 是 |
| 2 | `Series/桐城派名家文集/桐城派名家文集 10 薛福成集.pdf` | 桐城派名家文集 10 薛福成集.pdf | 桐城派名家文集 10 薛福成集 |  | 是 |
| 3 | `Series/桐城派名家文集/桐城派名家文集 11 姚永朴集 姚永概集.pdf` | 桐城派名家文集 11 姚永朴集 姚永概集.pdf | 桐城派名家文集 11 姚永朴集 姚永概集 |  | 是 |
| 4 | `Series/桐城派名家文集/桐城派名家文集 12 戴名世选集 方苞选集.pdf` | 桐城派名家文集 12 戴名世选集 方苞选集.pdf | 桐城派名家文集 12 戴名世选集 方苞选集 |  | 是 |
| 5 | `Series/桐城派名家文集/桐城派名家文集 13 刘大櫆选集 姚鼐选集 梅曾亮选集.pdf` | 桐城派名家文集 13 刘大櫆选集 姚鼐选集 梅曾亮选集.pdf | 桐城派名家文集 13 刘大櫆选集 姚鼐选集 梅曾亮选集 |  | 是 |
| 6 | `Series/桐城派名家文集/桐城派名家文集 14 曾国藩选集 张裕钊选集 黎庶昌选集.pdf` | 桐城派名家文集 14 曾国藩选集 张裕钊选集 黎庶昌选集.pdf | 桐城派名家文集 14 曾国藩选集 张裕钊选集 黎庶昌选集 |  | 是 |
| 7 | `Series/桐城派名家文集/桐城派名家文集 15 吴汝纶选集 贺涛选集 范当世选集.pdf` | 桐城派名家文集 15 吴汝纶选集 贺涛选集 范当世选集.pdf | 桐城派名家文集 15 吴汝纶选集 贺涛选集 范当世选集 |  | 是 |
| 8 | `Series/桐城派名家文集/桐城派名家文集 2 姚椿集.pdf` | 桐城派名家文集 2 姚椿集.pdf | 桐城派名家文集 2 姚椿集 |  | 是 |
| 9 | `Series/桐城派名家文集/桐城派名家文集 3 陈用光集.pdf` | 桐城派名家文集 3 陈用光集.pdf | 桐城派名家文集 3 陈用光集 |  | 是 |
| 10 | `Series/桐城派名家文集/桐城派名家文集 4 刘开集.pdf` | 桐城派名家文集 4 刘开集.pdf | 桐城派名家文集 4 刘开集 |  | 是 |
| 11 | `Series/桐城派名家文集/桐城派名家文集 5 管同集 吴敏树集.pdf` | 桐城派名家文集 5 管同集 吴敏树集.pdf | 桐城派名家文集 5 管同集 吴敏树集 |  | 是 |
| 12 | `Series/桐城派名家文集/桐城派名家文集 6 姚莹集.pdf` | 桐城派名家文集 6 姚莹集.pdf | 桐城派名家文集 6 姚莹集 |  | 是 |
| 13 | `Series/桐城派名家文集/桐城派名家文集 7 龙启瑞集 王拯集.pdf` | 桐城派名家文集 7 龙启瑞集 王拯集.pdf | 桐城派名家文集 7 龙启瑞集 王拯集 |  | 是 |
| 14 | `Series/桐城派名家文集/桐城派名家文集 8 马其昶集 戴钧衡集.pdf` | 桐城派名家文集 8 马其昶集 戴钧衡集.pdf | 桐城派名家文集 8 马其昶集 戴钧衡集 |  | 是 |
| 15 | `Series/桐城派名家文集/桐城派名家文集 9 方宗诚集.pdf` | 桐城派名家文集 9 方宗诚集.pdf | 桐城派名家文集 9 方宗诚集 |  | 是 |

## Albums/（12 行）

| # | 原路径 | 文件名 | 标题 | 作者 | 父目录在盘 |
|---:|---|---|---|---|---|
| 1 | `Albums/Vasulka Archive/urls.txt` | urls.txt |  |  | 否 |
| 2 | `Albums/Vasulka Archive/vasulka.org_archive_Artists4_Nygren,Scott_corresp.pdf` | vasulka.org_archive_Artists4_Nygren,Scott_corresp.pdf | vasulka.org_archive_Artists4_Nygren,Scott_corresp |  | 否 |
| 3 | `Albums/Vasulka Archive/vasulka.org_archive_Artists4_Nygren,Scott_general.pdf` | vasulka.org_archive_Artists4_Nygren,Scott_general.pdf | vasulka.org_archive_Artists4_Nygren,Scott_general |  | 否 |
| 4 | `Albums/Vasulka Archive/vasulka.org_archive_ExhONE_CentreGeorgesPomp_CGP003.pdf` | vasulka.org_archive_ExhONE_CentreGeorgesPomp_CGP003.pdf | vasulka.org_archive_ExhONE_CentreGeorgesPomp_CGP003 |  | 否 |
| 5 | `Albums/Vasulka Archive/vasulka.org_archive_ExhONE_CentreGeorgesPomp_CGPcat.pdf` | vasulka.org_archive_ExhONE_CentreGeorgesPomp_CGPcat.pdf | vasulka.org_archive_ExhONE_CentreGeorgesPomp_CGPcat |  | 否 |
| 6 | `Albums/Vasulka Archive/vasulka.org_archive_ExhONE_CentreGeorgesPomp_general.pdf` | vasulka.org_archive_ExhONE_CentreGeorgesPomp_general.pdf | vasulka.org_archive_ExhONE_CentreGeorgesPomp_general |  | 否 |
| 7 | `Albums/Vasulka Archive/vasulka.org_archive_Tools_show-fest3_ArsElectronica_92_glossary_Putney.pdf` | vasulka.org_archive_Tools_show-fest3_ArsElectronica_92_glossary_Putney.pdf | vasulka.org_archive_Tools_show-fest3_ArsElectronica_92_glossary_Putney |  | 否 |
| 8 | `Albums/Vasulka Archive/vasulka.org_archive_Tools_show-fest3_ArsElectronica_92_glossary_clip.pdf` | vasulka.org_archive_Tools_show-fest3_ArsElectronica_92_glossary_clip.pdf | vasulka.org_archive_Tools_show-fest3_ArsElectronica_92_glossary_clip |  | 否 |
| 9 | `Albums/Vasulka Archive/vasulka.org_archive_Tools_show-fest3_ArsElectronica_92_glossary_color-color.pdf` | vasulka.org_archive_Tools_show-fest3_ArsElectronica_92_glossary_color-color.pdf | vasulka.org_archive_Tools_show-fest3_ArsElectronica_92_glossary_color-color |  | 否 |
| 10 | `Albums/Vasulka Archive/vasulka.org_archive_show-fest3_ArsElectronica_92_glossary_Putney.pdf` | vasulka.org_archive_show-fest3_ArsElectronica_92_glossary_Putney.pdf | vasulka.org_archive_show-fest3_ArsElectronica_92_glossary_Putney |  | 否 |
| 11 | `Albums/Vasulka Archive/vasulka.org_archive_show-fest3_ArsElectronica_92_glossary_clip.pdf` | vasulka.org_archive_show-fest3_ArsElectronica_92_glossary_clip.pdf | vasulka.org_archive_show-fest3_ArsElectronica_92_glossary_clip |  | 否 |
| 12 | `Albums/Vasulka Archive/vasulka.org_archive_show-fest3_ArsElectronica_92_glossary_color-color.pdf` | vasulka.org_archive_show-fest3_ArsElectronica_92_glossary_color-color.pdf | vasulka.org_archive_show-fest3_ArsElectronica_92_glossary_color-color |  | 否 |

## Ernst Bloch/（1 行）

| # | 原路径 | 文件名 | 标题 | 作者 | 父目录在盘 |
|---:|---|---|---|---|---|
| 1 | `Ernst Bloch/[Historical Materialism Book Series 202] Cat Noir - Ernst Bloch’s Speculative Materialism_ Ontology, Epistem` | [Historical Materialism Book Series 202] Cat Noir - Ernst Bloch’s Speculative Materialism_ Ontology, Epistemology, Polit | [Historical Materialism Book Series 202] Cat Noir - Ernst Bloch’s Speculative Materialism_ Ontology, Epistemology, Polit | libgen.lc | 否 |