import re
import math
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

POLICY_KEYWORDS = {
    "objective","goal","strategy","policy","target","priority","increase",
    "improve","develop","promote","ensure","provide","implement","establish",
    "strengthen","achieve","reduce","support","national","government","ministry",
    "programme","plan","framework","production","quality","safety","sustainable",
    "sector","industry","stakeholder","requirement","standard","regulation",
    "export","import","finance","investment","research","development","welfare",
}

def ensure_nltk():
    for path, name in [
        ("tokenizers/punkt","punkt"),
        ("tokenizers/punkt_tab","punkt_tab"),
        ("corpora/stopwords","stopwords"),
    ]:
        try: nltk.data.find(path)
        except LookupError: nltk.download(name, quiet=True)

def clean_text(text: str) -> str:
    text = re.sub(r"[^\x00-\x7F]+"," ", text)
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$"," ", text)
    text = re.sub(r"[ \t]+"," ", text)
    text = re.sub(r"\n{3,}","\n\n", text)
    return text.strip()

def is_valid(sent: str) -> bool:
    words = sent.split()
    if len(words) < 10: return False
    alpha = [c for c in sent if c.isalpha()]
    if not alpha: return False
    if sum(1 for c in alpha if c.isupper())/len(alpha) > 0.45: return False
    for pat in [r"^actions?\s*:",r"^policy objective\s+\d",r"^[a-z]\)\s",r"^\d+\.\s+[A-Z\s]+$"]:
        if re.match(pat, sent.strip(), re.IGNORECASE): return False
    return True

def clean_sent(s: str) -> str:
    s = re.sub(r"^\s*[a-z]\)\s+","",s)
    s = re.sub(r"^\s*\d+\.\s+","",s)
    s = re.sub(r"^\s*[-–—]\s+","",s)
    return s.strip()

def tfidf_scores(sents, stops):
    n = len(sents); tf = {}; df = {}
    for s in sents:
        toks = word_tokenize(s.lower()); seen = set()
        for t in toks:
            if not t.isalpha() or t in stops or len(t)<3: continue
            tf[t] = tf.get(t,0)+1
            if t not in seen: df[t]=df.get(t,0)+1; seen.add(t)
    tfidf = {w: (tf[w])*(math.log((n+1)/(df.get(w,1)+1))+1) for w in tf}
    mx = max(tfidf.values()) if tfidf else 1
    return {w:v/mx for w,v in tfidf.items()}

def sent_score(sent, tfidf, stops, idx, total):
    toks = word_tokenize(sent.lower())
    ct = [t for t in toks if t.isalpha() and t not in stops and len(t)>=3]
    if not ct: return 0.0
    ts = sum(tfidf.get(t,0) for t in ct)/len(ct)
    kw = min(sum(1 for k in POLICY_KEYWORDS if k in sent.lower())*0.025, 0.25)
    n = len(sent.split())
    ls = 0.1 if 15<=n<=40 else 0.05 if 10<=n<15 or 40<n<=60 else 0.0
    ratio = idx/max(total-1,1)
    ps = 0.15 if ratio<0.10 else 0.10 if ratio<0.25 else 0.05 if ratio<0.50 else 0.0
    return 0.50*ts + 0.25*kw + 0.15*ps + 0.10*ls

def vec(sent, stops):
    toks = word_tokenize(sent.lower()); v = {}
    for t in toks:
        if t.isalpha() and t not in stops and len(t)>=3:
            v[t] = v.get(t,0)+1
    return v

def cosine(a, b):
    c = set(a)&set(b)
    if not c: return 0.0
    d = sum(a[w]*b[w] for w in c)
    na = math.sqrt(sum(x**2 for x in a.values()))
    nb = math.sqrt(sum(x**2 for x in b.values()))
    return d/(na*nb) if na and nb else 0.0

def mmr_select(scored, valid_sents, stops, n, lam=0.65):
    selected=[]; sel_vecs=[]; remaining=list(scored)
    while len(selected)<n and remaining:
        best=None; best_mmr=float("-inf")
        for sent,score in remaining:
            v=vec(sent,stops)
            max_sim=max((cosine(v,sv) for sv in sel_vecs), default=0.0)
            mmr=lam*score-(1-lam)*max_sim
            if mmr>best_mmr: best_mmr=mmr; best=(sent,score)
        if best:
            selected.append(best[0]); sel_vecs.append(vec(best[0],stops))
            remaining.remove(best)
    sent_set=set(selected)
    return [s for s in valid_sents if s in sent_set]

def extractive_summarise(text: str, num_sentences: int=10) -> str:
    ensure_nltk()
    cleaned = clean_text(text)
    all_sents = sent_tokenize(cleaned)
    valid = [s for s in all_sents if is_valid(s)]
    if len(valid)<=num_sentences:
        return " ".join(clean_sent(s) for s in valid)
    stops = set(stopwords.words("english"))
    tfidf = tfidf_scores(valid, stops)
    total = len(valid)
    scored = sorted(
        [(s, sent_score(s,tfidf,stops,i,total)) for i,s in enumerate(valid)],
        key=lambda x: x[1], reverse=True
    )
    selected = mmr_select(scored, valid, stops, num_sentences)
    return " ".join(clean_sent(s) for s in selected)

def extract_key_metadata(text: str) -> dict:
    ensure_nltk()
    cleaned = clean_text(text)
    sents = sent_tokenize(cleaned)
    words = word_tokenize(cleaned)
    content_words = [w for w in words if w.isalpha()]
    SKIP = {"vision","mission","goal","dairying for","create a competitive",
            "profitable ,sustainable","healthy and prosperous"}
    objectives = []
    for s in sents:
        stripped = s.strip(); lower = stripped.lower()
        if len(stripped.split())<6: continue
        if any(kw in lower for kw in SKIP): continue
        if re.match(r"^to\s+[a-z]", lower) and len(stripped.split()) <= 20:
            obj = re.sub(r"^\d+\.\s*","",stripped)
            obj = re.sub(r"^[a-zA-Z]\.\s*","",obj)
            obj = re.sub(r'\s+\d+\.\s*$', '', obj).strip()
            objectives.append(obj)

    return {
        "word_count": len(content_words),
        "sentence_count": len(sents),
        "detected_objectives": objectives[:12],
    }

def auto_detect_name(text: str, filename: str="") -> str:
    
    if filename:
        name = filename
        for ext in [".pdf",".txt",".docx",".PDF",".TXT"]:
            name = name.replace(ext,"")
        name = re.sub(r"[_\-]+"," ", name).strip().title()
        if len(name.split())>=2: return name

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for line in lines[:15]:
        if len(line.split())<3 or len(line)>100: continue
        if re.match(r"^\d",line): continue
        # Skip lines that look like section headings with only uppercase
        alpha = [c for c in line if c.isalpha()]
        if alpha and sum(1 for c in alpha if c.isupper())/len(alpha)>0.6: continue
        return re.sub(r"\s+"," ",line).strip().title()[:80]
    return "Uploaded Policy Document"
