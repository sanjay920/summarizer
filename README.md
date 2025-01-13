# GPT-4o Large Document Preservation

A command-line utility that **preserves and consolidates** extremely large text or Markdown files using [OpenAI’s gpt-4o model](https://platform.openai.com/docs/models). Unlike typical summarizers, this script maximizes information retention, maintaining technical specs, lists, tables, relationships, and more.

## Features

1. **Hierarchical Summaries**  
   - Automatically splits (or "chunks") your file into manageable pieces.  
   - **Parallel chunk processing** drastically speeds up large document handling.  
   - Recursively merges partial summaries until you get a single cohesive result.

2. **High-Fidelity Preservation**  
   - Emphasizes retention of exact numbers, quotes, relationships, and configuration details.  
   - Encourages minimal generalization to avoid discarding important specifics.

3. **Markdown-Compatible Output**  
   - The final pass uses markdown formatting for clarity, enabling easy reading or further editing.

---

## Installation

1. **Python 3.8+** (3.9+ recommended)
2. **Install Dependencies**:

   ```bash
   pip install openai tiktoken
   ```

3. **Set Up Your OpenAI API Key**:

   ```bash
   export OPENAI_API_KEY="sk-xxxxxx..."
   ```

   If your key is missing, the script will exit with an error.

4. **Clone or Download** the `summarize.py` file.

---

## Usage

```bash
python summarize.py <input_file> [options]
```

- `<input_file>` must be a `.txt` or `.md` file.

### Command-Line Options

- **`--output, -o <PATH>`**: Write the final output to `<PATH>`. If omitted, prints to terminal.  
- **`--model <MODEL>`**: OpenAI model to use (default `gpt-4o`).  
- **`--workers <N>`**: Number of parallel workers/threads (default `4`).  
- **`--verbose, -v`**: Verbose logging, including debug info on chunk counts and final merges.  
- **`--chunk-size <N>`**: Maximum tokens per chunk. Defaults to (max_context - max_output - overhead).

### Examples

1. **Basic Command**  

   ```bash
   python summarize.py my_report.txt
   ```

   Prints consolidated retention to stdout.

2. **Save to a File**  

   ```bash
   python summarize.py data_sheet.md -o data_summary.md
   ```

   Outputs a single final markdown file.

3. **Custom Chunk Size**  

   ```bash
   python summarize.py large_document.txt --chunk-size 50000
   ```

   Processes document in 50k token chunks.

4. **Parallel + Verbose**  

   ```bash
   python summarize.py large_document.txt --workers 8 --verbose
   ```

   Increases concurrency. Prints debug logs about chunking and summarization steps.

## What It Does

1. **Chunking**  
   Uses [tiktoken](https://github.com/openai/tiktoken) to split your text into safe segments. By default, chunks are sized to maximize GPT-4o's 128k token limit while leaving room for output and overhead. You can override this with `--chunk-size`.

2. **Parallel Summaries**  
   Processes multiple chunks in parallel, merging them into progressively smaller sets of summaries until a single cohesive text remains.

3. **Detailed Final Output**  
   - Preserves numeric values, tables, quotes, and hierarchical structures.  
   - Always attempts to keep the final result in a clean markdown format for easy reading or editing.

---

## Troubleshooting

1. **Missing `OPENAI_API_KEY`**  
   The script will exit with an error. Set it like this:

   ```bash
   export OPENAI_API_KEY="sk-xxxxxx..."
   ```

2. **Rate Limits or Many Parallel Workers**  
   If you encounter rate limit errors, reduce `--workers` or request a higher rate limit from OpenAI.

3. **Invalid JSON in Structure Analysis**  
   If the script's attempt to parse the JSON-like output for structure analysis fails, it will fallback gracefully. You will still get a final summary, but without advanced topic relationships.

4. **Installation Issues**  
   Make sure you have installed:

   ```bash
   pip install openai tiktoken
   ```
