import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rag import IndustrialRAG

def main():
    rag = IndustrialRAG()
    rag.load_index()

    print("AI-Powered Industrial Chatbot")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not question:
            continue

        try:
            answer, contexts = rag.answer(question)
            print("\nAssistant:", answer)
            print("\nRetrieved sources:")
            for c in contexts:
                print(f"- {c['source']} ({c['section']}, similarity={c['score']:.3f})")
            print()
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()
