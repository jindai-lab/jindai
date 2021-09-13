import re

RE_DIGITS = re.compile(r'[\+\-]?\d+')

def _opr(k):
    oprname = {
        'lte': 'le',
        'gte': 'ge',
        '': 'eq'
    }.get(k, k)
    return '__' + oprname + '__'


def _getattr(o, k, default=None):
    if '.' in k:
        for k_ in k.split('.'):
            o = _getattr(o, k_, default)
        return o

    if isinstance(o, dict):
        return o.get(k, default)
    elif isinstance(o, list) and RE_DIGITS.match(k):
        return o[int(k)] if 0 <= int(k) < len(o) else default
    else:
        return getattr(o, k, default)

    
def _hasattr(o, k):
    if '.' in k:
        for k_ in k.split('.')[:-1]:
            o = _getattr(o, k_)
        k = k.split('.')[-1]

    if isinstance(o, dict):
        return k in o
    elif isinstance(o, list) and RE_DIGITS.match(k):
        return 0 <= int(k) < len(o)
    else:
        return hasattr(o, k)


def _test_inputs(inputs, v, k='eq'):
    oprname = _opr(k)
    if oprname == '__in__':
        return inputs in v
    elif oprname == '__size__':
        return len(inputs) == v
    
    if isinstance(inputs, list):
        rr = False
        for v_ in inputs:
            rr = rr or _getattr(v_, oprname)(v)
            if rr:
                break
    else:
        rr = _getattr(inputs, oprname)(v)
    return rr is True


def execute_query_expr(parsed, inputs):
    r = True
    assert isinstance(parsed, dict), 'QueryExpr should be parsed first and well-formed.'
    for k, v in parsed.items():
        if k.startswith('$'):
            if k == '$and':
                rr = True
                for v_ in v:
                    rr = rr and execute_query_expr(v_, inputs)
            elif k == '$or':
                rr = False
                for v_ in v:
                    rr = rr or execute_query_expr(v_, inputs)
                    if rr:
                        break
            elif k == '$regex':
                rr = re.search(v, inputs) is not None
            elif k == '$options':
                continue
            else:
                rr = _test_inputs(inputs, v, k[1:])
            r = r and rr
        elif not isinstance(v, dict) or not [1 for v_ in v if v_.startswith('$')]:
            r = r and _test_inputs(_getattr(inputs, k), v)
        else:
            r = r and execute_query_expr(v, _getattr(inputs, k) if _hasattr(inputs, k) else None)
    return r

language_iso639 = dict([('ab', 'Abkhaz'),
('aa', 'Afar'),
('af', 'Afrikaans'),
('ak', 'Akan'),
('sq', 'Albanian'),
('am', 'Amharic'),
('ar', 'Arabic'),
('an', 'Aragonese'),
('hy', 'Armenian'),
('as', 'Assamese'),
('av', 'Avaric'),
('ae', 'Avestan'),
('ay', 'Aymara'),
('az', 'Azerbaijani'),
('bm', 'Bambara'),
('ba', 'Bashkir'),
('eu', 'Basque'),
('be', 'Belarusian'),
('bn', 'Bengali'),
('bh', 'Bihari'),
('bi', 'Bislama'),
('bs', 'Bosnian'),
('br', 'Breton'),
('bg', 'Bulgarian'),
('my', 'Burmese'),
('ca', 'Catalan'),
('ch', 'Chamorro'),
('ce', 'Chechen'),
('ny', 'Chichewa'),
('zh', 'Chinese'),
('cv', 'Chuvash'),
('kw', 'Cornish'),
('co', 'Corsican'),
('cr', 'Cree'),
('hr', 'Croatian'),
('cs', 'Czech'),
('da', 'Danish'),
('dv', 'Divehi'),
('nl', 'Dutch'),
('dz', 'Dzongkha'),
('en', 'English'),
('eo', 'Esperanto'),
('et', 'Estonian'),
('ee', 'Ewe'),
('fo', 'Faroese'),
('fj', 'Fijian'),
('fi', 'Finnish'),
('fr', 'French'),
('ff', 'Fula'),
('gl', 'Galician'),
('ka', 'Georgian'),
('de', 'German'),
('el', 'Greek'),
('gn', 'Guaraní'),
('gu', 'Gujarati'),
('ht', 'Haitian'),
('ha', 'Hausa'),
('he', 'Hebrew'),
('hz', 'Herero'),
('hi', 'Hindi'),
('ho', 'Hiri Motu'),
('hu', 'Hungarian'),
('ia', 'Interlingua'),
('id', 'Indonesian'),
('ie', 'Interlingue'),
('ga', 'Irish'),
('ig', 'Igbo'),
('ik', 'Inupiaq'),
('io', 'Ido'),
('is', 'Icelandic'),
('it', 'Italian'),
('iu', 'Inuktitut'),
('ja', 'Japanese'),
('jv', 'Javanese'),
('kl', 'Kalaallisut'),
('kn', 'Kannada'),
('kr', 'Kanuri'),
('ks', 'Kashmiri'),
('kk', 'Kazakh'),
('km', 'Khmer'),
('ki', 'Kikuyu'),
('rw', 'Kinyarwanda'),
('ky', 'Kirghiz'),
('kv', 'Komi'),
('kg', 'Kongo'),
('ko', 'Korean'),
('ku', 'Kurdish'),
('kj', 'Kwanyama'),
('la', 'Latin'),
('lb', 'Luxembourgish'),
('lg', 'Luganda'),
('li', 'Limburgish'),
('ln', 'Lingala'),
('lo', 'Lao'),
('lt', 'Lithuanian'),
('lu', 'Luba-Katanga'),
('lv', 'Latvian'),
('gv', 'Manx'),
('mk', 'Macedonian'),
('mg', 'Malagasy'),
('ms', 'Malay'),
('ml', 'Malayalam'),
('mt', 'Maltese'),
('mi', 'Māori'),
('mr', 'Marathi'),
('mh', 'Marshallese'),
('mn', 'Mongolian'),
('na', 'Nauru'),
('nv', 'Navajo'),
('nb', 'Norwegian Bokmål'),
('nd', 'North Ndebele'),
('ne', 'Nepali'),
('ng', 'Ndonga'),
('nn', 'Norwegian Nynorsk'),
('no', 'Norwegian'),
('ii', 'Nuosu'),
('nr', 'South Ndebele'),
('oc', 'Occitan'),
('oj', 'Ojibwe'),
('cu', 'Old Church Slavonic'),
('om', 'Oromo'),
('or', 'Oriya'),
('os', 'Ossetian, Ossetic'),
('pa', 'Panjabi, Punjabi'),
('pi', 'Pāli'),
('fa', 'Persian'),
('pl', 'Polish'),
('ps', 'Pashto, Pushto'),
('pt', 'Portuguese'),
('qu', 'Quechua'),
('rm', 'Romansh'),
('rn', 'Kirundi'),
('ro', 'Romanian, Moldavan'),
('ru', 'Russian'),
('sa', 'Sanskrit'),
('sc', 'Sardinian'),
('sd', 'Sindhi'),
('se', 'Northern Sami'),
('sm', 'Samoan'),
('sg', 'Sango'),
('sr', 'Serbian'),
('gd', 'Scottish Gaelic'),
('sn', 'Shona'),
('si', 'Sinhala'),
('sk', 'Slovak'),
('sl', 'Slovene'),
('so', 'Somali'),
('st', 'Southern Sotho'),
('es', 'Spanish'),
('su', 'Sundanese'),
('sw', 'Swahili'),
('ss', 'Swati'),
('sv', 'Swedish'),
('ta', 'Tamil'),
('te', 'Telugu'),
('tg', 'Tajik'),
('th', 'Thai'),
('ti', 'Tigrinya'),
('bo', 'Tibetan'),
('tk', 'Turkmen'),
('tl', 'Tagalog'),
('tn', 'Tswana'),
('to', 'Tonga'),
('tr', 'Turkish'),
('ts', 'Tsonga'),
('tt', 'Tatar'),
('tw', 'Twi'),
('ty', 'Tahitian'),
('ug', 'Uyghur'),
('uk', 'Ukrainian'),
('ur', 'Urdu'),
('uz', 'Uzbek'),
('ve', 'Venda'),
('vi', 'Vietnamese'),
('vo', 'Volapük'),
('wa', 'Walloon'),
('cy', 'Welsh'),
('wo', 'Wolof'),
('fy', 'Western Frisian'),
('xh', 'Xhosa'),
('yi', 'Yiddish'),
('yo', 'Yoruba'),
('za', 'Zhuang'),
('zu', 'Zulu'),
# non-ISO, used only for simplicity
('chs', 'Chinese Simplieifed'),
('cht', 'Chinese Traditional')
])