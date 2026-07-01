import os
import fitz
from bs4 import BeautifulSoup


def load_pdf(file_path):
    """
    Read PDF and return page-wise text.
    """
    pages = []

    pdf = fitz.open(file_path)

    for page_number, page in enumerate(pdf):
        text = page.get_text()

        if text.strip():
            pages.append({
                "text": text,
                "source": os.path.basename(file_path),
                "page": page_number + 1
            })

    pdf.close()

    return pages


def load_html(file_path):
    """
    Read HTML and return cleaned text.
    """

    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    text = soup.get_text(separator="\n")

    return [{
        "text": text,
        "source": os.path.basename(file_path),
        "page": 1
    }]


def load_documents(folder):
    """
    Load all PDFs and HTML files.
    """

    documents = []

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        if file.lower().endswith(".pdf"):
            documents.extend(load_pdf(path))

        elif file.lower().endswith(".html"):
            documents.extend(load_html(path))

    return documents


if __name__ == "__main__":

    docs = load_documents("documents")

    print(f"Loaded {len(docs)} document pages")

    for d in docs:
        print("-" * 50)
        print("Source :", d["source"])
        print("Page   :", d["page"])
        print(d["text"][:200])