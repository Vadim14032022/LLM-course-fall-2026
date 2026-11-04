"""Deterministic search representation; original LaTeX stays untouched."""
import re
import unicodedata

SEARCH_NORMALIZER_VERSION = 1
SUPERSCRIPTS = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺', '0123456789-+')


def normalize_search_text(text):
    # Preserve the exponent boundary before NFKC turns superscripts into digits.
    text = re.sub(r'(?<=[\w)\]}])[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+',
                  lambda m: '^{'+m[0].translate(SUPERSCRIPTS)+'}', text)
    text = unicodedata.normalize('NFKC', text).replace('−','-').replace('∣','|')
    text = text.replace('′', "'").replace('″', "''").replace('‴', "'''").replace('’', "'")
    text = re.sub(r'\^\s*\{\s*((?:\\prime\s*)+)\}', lambda m: "'"*m[1].count('\\prime'), text)
    text = re.sub(r'\^\s*\\prime\b', "'", text)
    text = re.sub(r'\\(?:bar|overline)\s*\{\s*([A-Za-z])\s*\}', lambda m: '\\bar '+m[1], text)

    def derivative(match):
        base, marks = match[1], match[2]
        name = base.replace('\\','cmd_')
        return base+' __math_derivative_'+name+'_'+str(len(marks))+' '
    # Restrict unadorned names to single math letters; do not treat English
    # contractions or another symbol's prime as the derivative of f.
    text = re.sub(r"(?<![\w\\])([A-Za-zА-Яа-я]|\\[A-Za-z]+)\s*('+)", derivative, text)

    def power(match):
        base = match[1]
        exponent = next(v for v in match.groups()[1:] if v is not None).replace(' ','')
        number = int(exponent)
        sign = 'negative' if number < 0 else 'positive'
        name = base.replace('\\','cmd_')
        return base+' ^ '+str(number)+' __math_power_'+name+'_'+sign+'_'+str(abs(number))+' '
    text = re.sub(r'(\\[A-Za-z]+|[A-Za-zА-Яа-я0-9]+)\s*\^\s*(?:\{\s*([+-]?\s*\d+)\s*\}|\(\s*([+-]?\s*\d+)\s*\)|([+-]?\d+))',power,text)
    return text


def search_tokens(text):
    normalized = normalize_search_text(text)
    parts = re.findall(r"\\[a-zA-Z]+|[\w]+|'+|[=<>+\-∂∞^!|/]", normalized)
    return [p if p.startswith(('\\','__math_')) else p.lower().replace('ё','е') for p in parts]
