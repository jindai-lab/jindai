# authors 归一化的回滚资料（2026-10-11）

## 背景
`author_unify.py --apply` 分两次跑完：第一次 611 行、第二次 117 行。
**第二次把第一次的备份文件同名覆盖了**，所以 611 行的原始快照只剩 117 行。
（脚本已修：备份名改为 `authors_unify_backup_<YYYYmmdd-HHMMSS>.tsv`，不再互相覆盖。）

## 现存的精确快照
- `reports/authors_unify_backup_20261011.tsv`：117 行（第二次 apply 的 pre-image，精确）
- 早前几步的备份仍在：`authors_quality_backup_20261011.tsv`(151)、
  `authors_t2_backup_20261011.tsv`(47)、`authors_stray3_backup_20261011.tsv`(3)

## 第一次 apply 的 494 行：可回滚性
- **C1/D1/X1 共 45 行**（把译者换成著者 / 把 `未知` 补成真名）——pre-image 逐条列在下面，
  按 path 找行即可原样写回。
- **U1 87 行**（异写合并）——在 117 行快照里，精确可回滚。
- **M1 约 362 行**（译名→原名，同一个人只换写法）——pre-image = 下表「原名 → 译名键」反查；
  同一原名对应多个译名键的是歧义项，已单列。
- **没有备份覆盖的只有 C1/D1/X1 那 45 行的原始组合**，逐条如下：

## C1/D1/X1 原始值（写回即可还原）
['白锡堃','György Lukács','本泽勒'] -> ['György Lukács']
['舒湘芹','Felix Klein'] -> ['Felix Klein']
['廖世德','Jiddu Krishnamurti'] -> ['Jiddu Krishnamurti']
['李绍明','Stuart Kauffman','徐彬'] -> ['Stuart Kauffman']
['Robert Francis Murphy','墨菲'] -> ['Robert Francis Murphy']
['Georgi Plekhanov','王荫庭'] -> ['Georgi Plekhanov']
['刘冉','George Park'] -> ['George Park']
['祝朝伟','Francis Aidan Gasquet'] -> ['Francis Aidan Gasquet']
['王春辰','Nigel Wentworth','董宏宇'] -> ['Nigel Wentworth']
['谭平','Don Thompson'] -> ['Don Thompson']
['Vincent Persichetti','刘烈武'] -> ['Vincent Persichetti']
['Theodor W. Adorno','张峰'] -> ['Theodor W. Adorno']
['John Marenbon','马仁邦'] -> ['John Marenbon']
['Georges Bataille','汪民安'] -> ['Georges Bataille']
['Douglas Hofstadter','丹尼特'] -> ['Douglas Hofstadter']
['楼巍','Ludwig Wittgenstein'] -> ['Ludwig Wittgenstein']
['Arthur O. Lovejoy','高秉江','张传有'] -> ['Arthur O. Lovejoy']
['Michel Kaplan','卡普兰'] -> ['Michel Kaplan']
['Boris Leonidovich Pasternak','布歇'] -> ['Boris Leonidovich Pasternak']
['刘军','Johan Huizinga'] -> ['Johan Huizinga']
['矶野富士子','Owen Lattimore'] -> ['Owen Lattimore']
['马凡','Eric Hobsbawm'] -> ['Eric Hobsbawm']
['John A. Walker','Judy Attfield','周丹丹','易菲'] -> ['John A. Walker','Judy Attfield']
['麦奎尔','温德尔'] -> ['Denis McQuail']
['渠东','汲喆'] -> ['Émile Durkheim']
['马夫罗金','Мавродцн'] -> ['Vladimir Mavrodin']
['古德温','周晓东','郭金','罗蕾'] -> ['Jason Goodwin']
['雅克·朗西埃'] -> ['Jacques Rancière']   (x2 行)
['易碎的绝对'] -> ['Slavoj Žižek']
['斯拉沃热·齐泽克'] -> ['Slavoj Žižek']

## M1 反查表（原名 → 当时的译名键）
共 222 个原名；歧义（一个原名对多个译名键）21 个：
- Antonio Gramsci  ←  葛兰西 / 安东尼奥·葛兰西
- Boethius  ←  波爱修斯 / 波爱修
- Edward Said  ←  萨义德 / 爱德华·萨义德
- Emmanuel Levinas  ←  列维纳斯 / 伊曼努尔·列维纳斯
- Friedrich Nietzsche  ←  尼采 / 弗里德里希·尼采
- Félix Guattari  ←  瓜塔里 / 菲利克斯·加塔利
- György Lukács  ←  卢卡奇 / 格奥尔格·卢卡奇
- Hannah Arendt  ←  阿伦特 / 汉娜·阿伦特
- Henri Lefebvre  ←  列斐伏尔 / 亨利·列斐伏尔
- Jean-Paul Sartre  ←  萨特 / 让-保罗·萨特
- Louis P. Pojman  ←  路易斯·P·波伊曼 / 路罗斯·P·波伊曼
- Marshall McLuhan  ←  麦克卢汉 / 马歇尔·麦克卢汉
- Martin Heidegger  ←  海德格尔 / 马丁·海德格尔
- Michael Hardt  ←  哈特 / 迈克尔·哈特
- Michel de Certeau  ←  德·塞尔托 / 德赛尔托
- Pierre Bourdieu  ←  布尔迪厄 / 皮埃尔·布尔迪厄
- Simone de Beauvoir  ←  波伏娃 / 西蒙娜·德·波伏瓦
- Slavoj Žižek  ←  齐泽克 / 斯拉沃热·齐泽克
- Susan Sontag  ←  桑塔格 / 苏珊·桑塔格
- Theodor W. Adorno  ←  阿多诺 / 特奥多尔·W.阿多诺 / 西奥多·阿多诺
- Walter Benjamin  ←  本雅明 / 瓦尔特·本雅明
