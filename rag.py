# Terminal RAG for PDF Q&A (FAISS + SentenceTransformers + Ollama)
# This script answers questions about PDFs in data 4 folder, directly and concisely.

import os, re, sys, glob, argparse, requests, faiss, fitz
import numpy as np
from sentence_transformers import SentenceTransformer

# Model and chunking configuration
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA    = "gemma3:1b"
CHUNK, OVERLAP, TOPK = 900, 150, 20 

def pages(pdf): 
    # Extract text from each page of a PDF
    try:
        d = fitz.open(pdf)
        if d.is_encrypted: d.authenticate("")
        for i in range(len(d)):
            t = d.load_page(i).get_text("text") or ""
            t = re.sub(r"-\n","",t); t = re.sub(r"\n{2,}","\n",t).strip()
            yield i+1, t
    except Exception as e:
        print(f"[WARN] Could not read {pdf}: {e}")

def chunk(text, n=CHUNK, o=OVERLAP):
    # Split text into overlapping chunks for better retrieval
    if not text: return []
    s = re.split(r"(?<=[.!?])\s+", text); out=[]; cur=""
    for x in s:
        if not cur: cur=x
        elif len(cur)+1+len(x)<=n: cur+=" "+x
        else: out.append(cur.strip()); cur=(cur[-o:]+" "+x).strip() if o and len(cur)>o else x
    if cur: out.append(cur.strip())
    return out

def build_corpus(folder):
    # Build a dictionary of PDF name to list of (page, chunk) tuples
    pdf_chunks = {}
    for pdf in sorted(glob.glob(os.path.join(folder,"**","*.pdf"), recursive=True)):
        base = os.path.basename(pdf)
        for pnum, ptxt in pages(pdf):
            for i, c in enumerate(chunk(ptxt)):
                if base not in pdf_chunks:
                    pdf_chunks[base] = []
                pdf_chunks[base].append((pnum, c))
    if not pdf_chunks: sys.exit(f"No PDFs/text found under: {folder}")
    return pdf_chunks

def embedder():
    # Load the embedding model and get embedding dimension
    m = SentenceTransformer(EMB_MODEL)
    dim = m.encode(["probe"], normalize_embeddings=True).shape[1]
    return m, dim

def build_index(pdf_chunks, model, dim):
    # Build a separate FAISS index for each PDF
    pdf_indexes = {}
    for pdf, chunks in pdf_chunks.items():
        texts = [c for _,c in chunks]
        if not texts: continue
        X = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        idx = faiss.IndexFlatIP(dim); idx.add(X)
        pdf_indexes[pdf] = (idx, texts, [p for p,_ in chunks])
    return pdf_indexes

def retrieve_per_pdf(q, model, pdf_indexes, k=TOPK):
    # For each PDF, retrieve top-k relevant chunks
    pdf_hits = {}
    for pdf, (idx, texts, pages) in pdf_indexes.items():
        if len(texts) == 0: continue
        qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        D,I = idx.search(qv, min(k, len(texts)))
        hits = [(pdf, pages[r], texts[r]) for r in I[0]]
        pdf_hits[pdf] = hits
    return pdf_hits

def ask_ollama_per_pdf(question, context, pdf_name):
    # Ask the LLM for a direct answer using only the context from one PDF
    prompt = (
        "You are an expert EXTRACTIVE QA assistant for PDFs. Use ONLY the provided context from the specified PDF. "
        "Answer the user's question as directly and specifically as possible, using only the information from the PDF. "
        "Do NOT provide any summary, general overview, or extra information. Do not repeat the question. Do not explain your reasoning. Just give the direct answer. "
        f"If the answer is not in the context, say so. Always use the word 'PDF' (not 'document').\n\n"
        f"# Context (from PDF: {pdf_name}, lines start with [PDF: <name> • p.<page>]):\n{context}\n"
    )
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": OLLAMA, "prompt": prompt, "stream": False}, timeout=120)
    r.raise_for_status()
    return r.json().get("response","").strip()

def answer(q, model, pdf_indexes):
    # Retrieve relevant chunks for the question from each PDF
    pdf_hits = retrieve_per_pdf(q, model, pdf_indexes, k=TOPK)
    answers = []
    cited = []
    for pdf, hits in pdf_hits.items():
        if not hits: continue
        ctx = "\n\n".join(f"[PDF: {pdf} • p.{p}] {t}" for _,p,t in hits)
        ans = ask_ollama_per_pdf(q, ctx, pdf)
        # Only include PDFs that actually provide a direct answer
        if ans.strip() and not ans.lower().startswith("no relevant context") and not ans.lower().startswith("the answer is not in the context"):
            answers.append(f"[PDF: {pdf}]\n{ans}")
            cited.append((pdf, sorted({p for _,p,_ in hits})))
    if not answers:
        print("\n--- Answer ---\nNo relevant context found.")
        return
    # Print answers and sources
    print("\n--- Answer ---\n" + "\n\n".join(answers))
    print("\n--- Sources ---")
    for pdf, pages in cited:
        page_list = ', '.join(str(p) for p in pages)
        print(f"- {pdf} (pages: {page_list})")

def main():
    # Parse arguments and build the RAG index
    ap = argparse.ArgumentParser(description="RAG for PDFs (terminal Q&A).")
    ap.add_argument("folder", nargs="?", default="data 4", help="PDF folder (default: 'data 4').")
    ap.add_argument("--ask", help="One-shot question (non-interactive).")
    args = ap.parse_args()

    print(f"[RAG] Indexing PDFs in: {args.folder}")
    pdf_chunks = build_corpus(args.folder)
    model, dim = embedder(); pdf_indexes = build_index(pdf_chunks, model, dim)
    print(f"[RAG] Ready. {sum(len(v[1]) for v in pdf_indexes.values())} chunks indexed from {len(pdf_indexes)} PDFs.")

    if args.ask:
        answer(args.ask, model, pdf_indexes)
        return

    print("\nWelcome to PDF RAG Q&A!")
    print("You can ask any question about the PDFs in the data 4 folder.")
    print("Type 'exit' to quit.\n")
    while True:
        try:
            q = input("Please enter your question: ").strip()
        except (EOFError, KeyboardInterrupt): print(); break
        if not q: continue
        if q.lower() in {"exit","quit","q",":q"}: break
        answer(q, model, pdf_indexes)

if __name__ == "__main__": main()
