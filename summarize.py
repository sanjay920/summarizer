#!/usr/bin/env python3
"""
summarize.py
------------

A command-line utility to comprehensively preserve and consolidate large text or
Markdown files using GPT-4o. It automatically chunks the document, processes
each chunk in parallel, and merges all retained information into a final,
high-density output—optionally focusing on a user-specified topic.

Usage (typical):
    python summarize.py path/to/file.md -o summary_output.txt --workers 8 --verbose
"""

import tiktoken
from typing import List
import concurrent.futures
import argparse
from pathlib import Path
import sys
import os


class LargeDocumentSummarizer:
    """
    Summarizes very large documents with hierarchical chunking using gpt-4o.
    Supports parallel calls to speed up summarization.
    Optionally uses a 'topic' for specialized focus and structure.
    """

    def __init__(
        self,
        client,
        model: str = "gpt-4o",
        max_context_tokens: int = 128000,
        max_output_tokens: int = 16384,
        overhead_tokens: int = 2000,
        max_chunk_tokens: int = None,
        max_workers: int = 4,
        verbose: bool = False,
    ):
        """
        :param client: An OpenAI() client instance (from openai import OpenAI).
        :param model: Model name (e.g., 'gpt-4o'), must be valid for encoding_for_model().
        :param max_context_tokens: Maximum context length for GPT-4o (default: 128000).
        :param max_output_tokens: Maximum tokens GPT-4o can generate (default: 16384).
        :param overhead_tokens: Token buffer for system/developer instructions, etc. (default: 2000).
        :param max_chunk_tokens: Maximum tokens per chunk (default: max_context_tokens - max_output_tokens - overhead_tokens).
        :param max_workers: Number of parallel threads for summarization calls (default: 4).
        :param verbose: Whether to print additional logs and progress information.
        """
        self.client = client
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.overhead_tokens = overhead_tokens
        self.max_workers = max_workers
        self.verbose = verbose

        # Load the tokenizer for the specified model
        self.enc = tiktoken.encoding_for_model(self.model)

        # Calculate or use provided max chunk size
        default_chunk_size = (
            self.max_context_tokens - self.max_output_tokens - self.overhead_tokens
        )
        self.max_chunk_size = (
            max_chunk_tokens if max_chunk_tokens is not None else default_chunk_size
        )

        if self.max_chunk_size <= 0:
            raise ValueError(
                "Calculated or provided max_chunk_size is non-positive. "
                "Adjust max_chunk_tokens or reduce overhead_tokens/max_output_tokens."
            )

        if self.verbose:
            print(f"[DEBUG] Using model: {self.model}")
            print(f"[DEBUG] max_context_tokens: {self.max_context_tokens}")
            print(f"[DEBUG] max_output_tokens: {self.max_output_tokens}")
            print(f"[DEBUG] overhead_tokens: {self.overhead_tokens}")
            print(f"[DEBUG] max_chunk_size: {self.max_chunk_size}")
            print(f"[DEBUG] max_workers: {self.max_workers}")

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into token-based chunks, ensuring each chunk fits within
        (max_context_tokens - overhead_tokens - max_output_tokens).
        """
        tokens = self.enc.encode(text)
        chunks = []

        if self.verbose:
            print(f"[DEBUG] Total tokens in document: {len(tokens)}")
            print("[DEBUG] Splitting into chunks...")

        for i in range(0, len(tokens), self.max_chunk_size):
            chunk_slice = tokens[i : i + self.max_chunk_size]
            chunk_text = self.enc.decode(chunk_slice)
            chunks.append(chunk_text)

        if self.verbose:
            print(f"[DEBUG] Created {len(chunks)} chunk(s).")

        return chunks

    def summarize_chunk(self, chunk: str) -> str:
        """
        Summarizes a single chunk using an intensive, detail-preserving prompt.
        """
        system_prompt = """You are an expert in information preservation and technical documentation.
Your task is to create a dense, detailed retention of the input content.

Critical rules:

1. PRESERVE ALL:
   - Technical specifications, numbers, and measurements
   - Names, identifiers, key terms
   - Procedural steps and sequences
   - Relationships and dependencies
   - Configuration details and parameters
   - Important direct quotes

2. Structure your response as:
   <METADATA>
   - Document type: (code/technical/narrative/documentation/other)
   - Key terms: [list important terms/identifiers]
   - Structure type: (hierarchical/sequential/reference/other)
   </METADATA>

   <CORE_CONTENT>
   [Detailed preservation of the content, maintaining original structure if possible]
   </CORE_CONTENT>

   <RELATIONSHIPS>
   [Dependencies, connections, cross-references found in the content]
   </RELATIONSHIPS>

3. Use direct quotes where precision matters
4. Maintain hierarchical structure if it exists
5. Preserve all numeric/technical data
6. Keep lists, tables, or structured data in original format if feasible"""

        user_prompt = f"""Analyze and preserve this content with maximum detail:

{chunk}

Remember:
- Maintain original structure
- Retain all numeric values
- Include complete lists/tables
- Use quotes for critical data
- Keep relationships and dependencies
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.max_output_tokens,
        )
        return response.choices[0].message.content.strip()

    def summarize_chunks_in_parallel(self, chunks: List[str]) -> List[str]:
        """
        Summarize multiple chunks in parallel using ThreadPoolExecutor.
        """
        summaries = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Dispatch summarization tasks
            future_to_chunk = {
                executor.submit(self.summarize_chunk, chunk): chunk for chunk in chunks
            }
            for future in concurrent.futures.as_completed(future_to_chunk):
                summaries.append(future.result())

        if self.verbose:
            print(f"[DEBUG] Summarized {len(chunks)} chunk(s) in parallel.")

        return summaries

    def summarize_document_once(self, text: str) -> List[str]:
        """
        Splits text into chunks and summarizes each chunk in parallel,
        returning a list of chunk-level summaries.
        """
        chunks = self.chunk_text(text)
        return self.summarize_chunks_in_parallel(chunks)

    def iterative_summarize(self, summaries: List[str]) -> str:
        """
        Recursively merges a list of chunk summaries until a single summary remains.
        If the final pass is detected, use a special final_reduction prompt.
        """
        if len(summaries) == 1:
            return summaries[0]

        if self.verbose:
            print(f"[DEBUG] Combining {len(summaries)} summaries into a new text...")

        combined_text = "\n\n".join(summaries)
        # Check if the newly combined text can fit in one chunk
        # If so, we proceed with final reduction
        if len(self.chunk_text(combined_text)) <= 1:
            if self.verbose:
                print("[DEBUG] Performing final reduction with special prompt...")
            return self.final_reduction(combined_text)

        # Otherwise, summarize again in parallel
        next_level_summaries = self.summarize_document_once(combined_text)
        return self.iterative_summarize(next_level_summaries)

    def final_reduction(self, text: str) -> str:
        """
        Produces a final, consolidated version of the retained information.
        Maintains maximum detail in a cohesive format.
        """
        system_prompt = """You are creating the final consolidated version of preserved information. 
Preserve maximum detail and maintain a cohesive structure.

Requirements:

1. DO NOT summarize away critical details
2. Keep ALL:
   - Technical specs, numeric values
   - Names and IDs
   - Procedural steps
   - Configuration details
   - Interrelationships

3. Use markdown for clarity
4. Preserve essential formatting
5. Keep direct quotes intact
"""

        user_prompt = f"""Consolidate the following retention text into a single, cohesive document, 
while preserving all critical information:

{text}

You must:
- Retain specificity
- Keep numeric values
- Use direct quotes where originally present
- Maintain references, relationships, and any structured data
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.max_output_tokens,
        )
        return response.choices[0].message.content.strip()

    def analyze_document_structure(self, document: str) -> dict:
        """
        Optional helper that tries to detect major topics, sections, and relationships
        from the first chunk of text. Returns a JSON-like structure as a Python dict.
        """
        system_prompt = """You are an expert document analyzer. Examine this content and identify:
1. Major topics and themes
2. Document structure and sections
3. Topic relationships and hierarchies

Return your analysis in this JSON-like structure:
{
    "main_topics": ["topic1", "topic2", ...],
    "sections": [
        {
            "content_type": "code|documentation|config|other",
            "topics": ["topic1", "topic2"],
            "relevance": "high|medium|low",
            "context_importance": "high|medium|low"
        }
    ],
    "topic_relationships": {
        "topic1": ["related_topic1", "related_topic2"]
    }
}"""

        # For analysis, we only use the first chunk of the document
        partial_text = document[: self.max_chunk_size]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": partial_text},
                ],
                max_tokens=2000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if self.verbose:
                print(f"[DEBUG] Document analysis failed: {e}")
            return {}

    def summarize_large_document(self, document: str) -> str:
        """
        Main entry point for summarization:
        1) Chunk & summarize in parallel
        2) Recursively merge until single summary remains
        3) Perform final reduction for a cohesive, detail-rich result
        """
        if self.verbose:
            print("[DEBUG] Starting multi-pass summarization...")

        # First pass: chunk & summarize
        chunks = self.chunk_text(document)
        summaries = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_chunk = {
                executor.submit(self.summarize_chunk, chunk): chunk for chunk in chunks
            }
            for future in concurrent.futures.as_completed(future_to_chunk):
                summaries.append(future.result())

        # Merge intermediate summaries
        intermediate_summary = self.iterative_summarize(summaries)

        # Final pass: produce cohesive result
        final_summary = self.final_reduction(intermediate_summary)
        return final_summary


def main():
    parser = argparse.ArgumentParser(
        description="Preserve and consolidate large text/markdown files using GPT-4o."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input file (.txt or .md)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output file (optional). If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose mode for debugging/log messages.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Maximum tokens per chunk. Defaults to (max_context - max_output - overhead).",
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        sys.exit(f"ERROR: Input file not found: {input_path}")
    if input_path.suffix.lower() not in [".txt", ".md"]:
        sys.exit("ERROR: Input file must be a .txt or .md file")

    # Read input file
    if args.verbose:
        print(f"[DEBUG] Reading input file: {input_path}")
    try:
        with input_path.open("r", encoding="utf-8") as f:
            document = f.read()
    except Exception as e:
        sys.exit(f"ERROR: Unable to read input file: {e}")

    # Check for OPENAI_API_KEY
    if "OPENAI_API_KEY" not in os.environ:
        sys.exit(
            "ERROR: OPENAI_API_KEY environment variable not found.\n"
            "Please set it before running the script, e.g.:\n\n"
            "  export OPENAI_API_KEY='sk-xxxxxxx'\n"
        )

    # Initialize OpenAI client
    try:
        from openai import OpenAI

        client = OpenAI()  # Uses OPENAI_API_KEY from environment
    except ImportError:
        sys.exit(
            "ERROR: openai package not found. Install via:\n\n" "  pip install openai\n"
        )
    except Exception as e:
        sys.exit(f"ERROR: Failed to initialize OpenAI client: {e}")

    # Create summarizer and process document
    summarizer = LargeDocumentSummarizer(
        client,
        model=args.model,
        max_chunk_tokens=args.chunk_size,
        max_workers=args.workers,
        verbose=args.verbose,
    )

    try:
        final_summary = summarizer.summarize_large_document(document)
    except Exception as e:
        sys.exit(f"ERROR: Summarization failed: {e}")

    # Handle output
    if args.output:
        output_path = Path(args.output)
        if args.verbose:
            print(f"[DEBUG] Writing summary to: {output_path}")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(final_summary, encoding="utf-8")
            print(f"Summary written to: {output_path}")
        except Exception as e:
            sys.exit(f"ERROR: Failed writing output file: {e}")
    else:
        print("\nFINAL CONSOLIDATED OUTPUT:")
        print("-" * 80)
        print(final_summary)
        print("-" * 80)


if __name__ == "__main__":
    main()
