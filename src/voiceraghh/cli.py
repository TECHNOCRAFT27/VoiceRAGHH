import argparse
from voiceraghh.pipeline import VoiceRAG


def main():
    parser = argparse.ArgumentParser(description="Voice RAG Pipeline")
    sub = parser.add_subparsers(dest="command")
    
    index_cmd = sub.add_parser("index", help="Build index from text file")
    index_cmd.add_argument("file", help="Text file to index")
    index_cmd.add_argument("-o", "--output", default="./data/index", help="Index output path")
    
    query_cmd = sub.add_parser("query", help="Query the index")
    query_cmd.add_argument("text", help="Query text")
    query_cmd.add_argument("-i", "--index", default="./data/index", help="Index path")
    
    voice_cmd = sub.add_parser("voice", help="Voice query")
    voice_cmd.add_argument("audio", help="Audio file path")
    voice_cmd.add_argument("-i", "--index", default="./data/index", help="Index path")
    
    args = parser.parse_args()
    
    if args.command == "index":
        with open(args.file) as f:
            texts = f.read().split("\n\n")
        rag = VoiceRAG()
        rag.build_index([t for t in texts if t.strip()])
        rag.vectorstore.save(args.output)
        print(f"Indexed {len(texts)} chunks to {args.output}")
    
    elif args.command == "query":
        rag = VoiceRAG(vectorstore_path=args.index)
        result = rag.answer(args.text)
        print(f"Answer: {result.answer}")
    
    elif args.command == "voice":
        rag = VoiceRAG(vectorstore_path=args.index)
        result = rag.voice_query(args.audio)
        print(f"Answer: {result.answer}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
