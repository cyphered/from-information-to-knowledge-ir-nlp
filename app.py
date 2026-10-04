import re
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="From Information to Knowledge", page_icon="🔎", layout="wide")

DOCUMENTS = [
    {"id":"d1","title":"Information Retrieval Foundations","text":"Information retrieval is the task of finding relevant information from a collection of documents. A retrieval system represents queries and documents, scores their relevance, and returns a ranked list."},
    {"id":"d2","title":"BM25 and Ranking","text":"BM25 is a probabilistic ranking model used in information retrieval. It considers term frequency, inverse document frequency, document length normalization, and tunable parameters when scoring documents."},
    {"id":"d3","title":"Natural Language Processing","text":"Natural language processing enables computers to process human language. Tokenization, normalization, stemming, lemmatization, part of speech tagging, and named entity recognition are common NLP tasks."},
    {"id":"d4","title":"Semantic Representations","text":"Dense vector representations encode text as points in a high dimensional space. Similar meanings can be represented by vectors that are close under a suitable similarity measure."},
    {"id":"d5","title":"Semantic Search","text":"Semantic search retrieves information by meaning rather than relying only on exact word overlap. Sentence embeddings and vector databases can support semantic retrieval."},
    {"id":"d6","title":"Question Answering","text":"Question answering systems retrieve evidence and produce an answer to a natural language question. Retrieval augmented generation combines retrieval with a language model to ground responses in documents."},
    {"id":"d7","title":"Information Extraction","text":"Information extraction converts unstructured text into structured facts. Named entity recognition can identify people, organizations, locations, dates, and other entity types."},
    {"id":"d8","title":"Knowledge Discovery","text":"Knowledge discovery involves identifying useful patterns and relationships in information. Extracted entities and relations can be organized into structured knowledge representations."},
    {"id":"d9","title":"Multilingual Information Access","text":"Multilingual information access allows users to search across languages. Multilingual language models can map text from languages such as English, Hindi, and Telugu into shared vector spaces."},
    {"id":"d10","title":"Evaluation of Retrieval","text":"Retrieval evaluation measures whether relevant documents appear near the top of a ranked list. Precision at k, recall, mean reciprocal rank, and normalized discounted cumulative gain are common metrics."},
    {"id":"d11","title":"Machine Learning for Search","text":"Machine learning can learn representations, ranking functions, or classifiers from data. Retrieval pipelines often combine deterministic lexical methods with learned dense representations."},
    {"id":"d12","title":"Algorithms and Data Structures","text":"Search systems rely on algorithms and data structures such as inverted indexes, hash tables, graphs, heaps, and vector indexes. Efficient indexing reduces the cost of ranking large document collections."},
    {"id":"d13","title":"सार्वजनिक डिजिटल जानकारी","text":"सूचना पुनर्प्राप्ति प्रणाली उपयोगकर्ता के प्रश्न के आधार पर प्रासंगिक दस्तावेज़ खोजती है। बहुभाषी मॉडल विभिन्न भाषाओं के पाठ को साझा वेक्टर स्थान में प्रस्तुत कर सकते हैं।"},
    {"id":"d14","title":"తెలుగు సమాచార ప్రాప్తి","text":"సమాచార పునరుద్ధరణ వ్యవస్థలు ప్రశ్నకు సంబంధించిన పత్రాలను కనుగొని ర్యాంక్ చేస్తాయి. బహుభాషా భాషా నమూనాలు వివిధ భాషల వచనాన్ని వెక్టర్ రూపంలో సూచించగలవు."},
]

@st.cache_data
def corpus():
    return DOCUMENTS

@st.cache_resource
def tfidf_index(texts):
    v = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2))
    X = v.fit_transform(texts)
    return v, X

def lexical_search(query, docs, top_k):
    texts=[d["text"] for d in docs]
    v,X=tfidf_index(texts)
    scores=cosine_similarity(v.transform([query]),X).ravel()
    order=np.argsort(-scores)[:top_k]
    return [(docs[i], float(scores[i])) for i in order if scores[i] > 0]

@st.cache_resource
def dense_index(texts, model_name):
    from sentence_transformers import SentenceTransformer
    import faiss
    model=SentenceTransformer(model_name)
    emb=np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype="float32")
    index=faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return model,index

def dense_search(query, docs, top_k, model_name):
    model,index=dense_index([d["text"] for d in docs], model_name)
    q=model.encode([query], normalize_embeddings=True)
    scores,ids=index.search(np.asarray(q,dtype="float32"), top_k)
    return [(docs[int(i)], float(s)) for s,i in zip(scores[0],ids[0]) if i >= 0]

def sentence_qa(question, docs):
    q=set(re.findall(r"\b[a-zA-Z]{3,}\b", question.lower()))
    candidates=[]
    for d in docs:
        for s in re.split(r"(?<=[.!?])\s+", d["text"]):
            words=set(re.findall(r"\b[a-zA-Z]{3,}\b", s.lower()))
            overlap=len(q & words)
            if overlap:
                candidates.append((overlap, s, d["title"]))
    return sorted(candidates, reverse=True)[:5]

def heuristic_entities(text):
    patterns = {
        "Email": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
        "URL": r"https?://\S+",
        "Year": r"\b(?:19|20)\d{2}\b",
        "Capitalized Phrase": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b",
    }
    return [(label,m) for label,pat in patterns.items() for m in re.findall(pat,text)]

def extract_entities(text):
    try:
        import spacy
        nlp=spacy.load("en_core_web_sm")
        return [(e.label_,e.text) for e in nlp(text).ents]
    except Exception:
        return heuristic_entities(text)

st.title("From Information to Knowledge")
st.caption("An IR + NLP laboratory connecting retrieval, ranking, semantic processing, information extraction, multilingual access, and question answering.")

with st.sidebar:
    st.header("Collection")
    uploaded=st.file_uploader("Add .txt documents", type=["txt"], accept_multiple_files=True)
    docs=corpus().copy()
    if uploaded:
        for f in uploaded:
            docs.append({"id":f.name,"title":f.name,"text":f.read().decode("utf-8", errors="ignore")})
        st.success(f"{len(uploaded)} custom document(s) added")
    top_k=st.slider("Top K results",1,10,5)
    model_name=st.selectbox("Dense model",[
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/all-MiniLM-L6-v2"
    ])
    st.markdown("**Methods:** TF-IDF, cosine similarity, FAISS, sentence embeddings, NER, evidence ranking.")

tab1,tab2,tab3,tab4=st.tabs(["🔎 Lexical Retrieval","🧠 Dense / Semantic Search","🧩 Information Extraction","❓ Question Answering"])

with tab1:
    st.subheader("Retrieval Models, Ranking and Evaluation")
    q=st.text_input("Search the collection", placeholder="e.g. How does semantic search work?")
    if q:
        results=lexical_search(q,docs,top_k)
        if results:
            for rank,(d,s) in enumerate(results,1):
                st.markdown(f"**{rank}. {d['title']}**  \nTF-IDF cosine score: `{s:.3f}`")
                st.write(d["text"])
                st.divider()
        else: st.info("No lexical overlap found. Try a broader query.")

with tab2:
    st.subheader("Language Representation and Semantic Processing")
    st.write("Dense retrieval encodes documents and queries as vectors, then uses FAISS inner-product search over normalized embeddings.")
    q2=st.text_input("Semantic query", placeholder="e.g. finding documents by meaning", key="dense")
    if q2:
        try:
            results=dense_search(q2,docs,top_k,model_name)
            for rank,(d,s) in enumerate(results,1):
                st.markdown(f"**{rank}. {d['title']}**  \nVector similarity: `{s:.3f}`")
                st.write(d["text"])
                st.divider()
        except Exception as e:
            st.error("Dense retrieval could not start. The first use needs the selected Hugging Face model to download.")
            st.code(str(e))

with tab3:
    st.subheader("Information Extraction and Knowledge Discovery")
    text=st.text_area("Paste text to extract entities", value="OpenAI researchers discussed information retrieval in Hyderabad in 2026. Contact research@example.org.")
    if st.button("Extract entities"):
        entities=extract_entities(text)
        if entities:
            for label,value in entities:
                st.write(f"**{label}:** {value}")
        else: st.info("No entities detected.")

with tab4:
    st.subheader("Question Answering")
    question=st.text_input("Ask a question about the collection", placeholder="What is BM25?")
    if question:
        answers=sentence_qa(question,docs)
        if answers:
            best=answers[0]
            st.success(best[1])
            st.caption(f"Evidence source: {best[2]} • lexical overlap score: {best[0]}")
            st.write("Other evidence:")
            for score,s,title in answers[1:]:
                st.write(f"- {s}  ({title}, overlap={score})")
        else: st.info("No matching evidence found.")

st.divider()
st.caption("Research note: compare classical lexical retrieval with learned dense retrieval and evaluate them with labelled relevance judgments.")
