# Writer Extraction and Management Feature

## Overview

The Writer Extraction and Management feature automatically extracts writer/author names from documents, tracks their information, and references all their writings across your document collection.

## Features

- **Automatic Writer Extraction**: Extracts writer names from documents using pattern matching
- **Fuzzy Matching**: Automatically matches similar writer names to avoid duplicates
- **Information Tracking**: Stores biographical information, achievements, and affiliations
- **Writing References**: Tracks all documents where a writer is mentioned
- **Automatic Summarization**: Generates summaries of each writer's works
- **Update on New Mentions**: Automatically updates writer information when new documents are added

## How It Works

### 1. Document Processing

When a document is uploaded to the `docs/` folder and processed:

1. The system extracts the full text from the document
2. Writer names are identified using pattern matching (supports English and Arabic)
3. For each writer found:
   - The system checks if the writer already exists (using fuzzy matching)
   - If new, a writer profile is created
   - If existing, the profile is updated with new information
4. Writer information is extracted from the context around their mentions
5. The document is added to the writer's list of writings

### 2. Writer Matching

The system uses intelligent fuzzy matching to identify when the same writer appears under different variations:

- Exact name matching
- Partial name matching (e.g., "John Smith" vs "J. Smith")
- Similarity scoring (configurable threshold: 85% by default)
- Alias support for manually adding name variations

### 3. Information Extraction

For each writer, the system extracts:

- **Mentions**: All sentences where the writer is mentioned
- **Context Snippets**: Contextual information about the writer
- **Achievements**: Awards, honors, and accomplishments
- **Affiliations**: Universities, organizations, institutions
- **Topics**: Fields of work or research areas
- **Biographical Summary**: Generated from collected information

## API Endpoints

### List All Writers

```http
GET /api/writers
```

Returns all writers in the database with their complete information.

**Response:**
```json
{
  "writers": [...],
  "total_count": 5,
  "timestamp": "2025-11-26T10:00:00"
}
```

### Get Writer Statistics

```http
GET /api/writers/stats
```

Returns statistics about the writer database.

**Response:**
```json
{
  "stats": {
    "total_writers": 5,
    "total_writings_tracked": 15,
    "total_mentions": 42,
    "top_writers": [
      {
        "name": "John Doe",
        "mentions": 12,
        "documents": 3
      }
    ]
  }
}
```

### Get Writer by ID

```http
GET /api/writers/{writer_id}
```

Get detailed information about a specific writer.

**Response:**
```json
{
  "writer": {
    "writer_id": "writer_john_doe_a1b2c3d4",
    "name": "John Doe",
    "aliases": ["J. Doe", "John D."],
    "created_at": "2025-11-26T09:00:00",
    "updated_at": "2025-11-26T10:00:00",
    "writings": [
      {
        "document": "paper1.pdf",
        "added_at": "2025-11-26T09:00:00",
        "mentions_count": 5
      }
    ],
    "total_mentions": 12,
    "information": {
      "bio_summary": "...",
      "achievements": ["Nobel Prize", "..."],
      "affiliations": ["MIT", "Stanford"],
      "topics": ["AI", "Machine Learning"],
      "context_snippets": ["..."]
    },
    "writings_summary": "John Doe is mentioned in 3 document(s)...",
    "metadata": {}
  }
}
```

### Search Writers

```http
POST /api/writers/search
```

Search for writers by name or information.

**Request Body:**
```json
{
  "query": "john"
}
```

**Response:**
```json
{
  "query": "john",
  "results": [...],
  "total_results": 2
}
```

### Get Writer by Name

```http
GET /api/writers/by-name/{name}
```

Get writer by name with fuzzy matching.

**Example:**
```http
GET /api/writers/by-name/John%20Doe
```

### Get Writers by Document

```http
GET /api/documents/{filename}/writers
```

Get all writers mentioned in a specific document.

**Example:**
```http
GET /api/documents/paper1.pdf/writers
```

**Response:**
```json
{
  "document": "paper1.pdf",
  "writers": [...],
  "total_writers": 3
}
```

### Add Writer Alias

```http
POST /api/writers/add-alias
```

Add an alias to a writer's profile.

**Request Body:**
```json
{
  "writer_id": "writer_john_doe_a1b2c3d4",
  "alias": "J. Doe"
}
```

### Merge Writers

```http
POST /api/writers/merge
```

Merge two writer entries (useful for handling duplicates).

**Request Body:**
```json
{
  "writer_id1": "writer_john_doe_a1b2c3d4",
  "writer_id2": "writer_j_doe_e5f6g7h8",
  "keep_id": "writer_john_doe_a1b2c3d4"
}
```

## Writer Name Extraction Patterns

The system looks for various patterns to identify writers:

### English Patterns

- "written by [Name]"
- "author: [Name]"
- "by [Name]"
- "Dr./Prof./Mr./Ms./Mrs. [Name]"
- "[Name] wrote/writes/published/authored"
- Capitalized names that appear multiple times

### Arabic Patterns

- "كتبه [Name]"
- "المؤلف: [Name]"
- "بقلم [Name]"
- "للكاتب [Name]"
- "تأليف [Name]"
- "د./أ./الدكتور/الأستاذ [Name]"

## Data Storage

Writer data is stored in the `writers_db/` folder:

- **writers_index.json**: Main index containing all writer information
- Each writer entry includes complete metadata, writings references, and extracted information

## System Status

The system status endpoint now includes writer statistics:

```http
GET /api/status
```

Returns comprehensive system status including writer database statistics.

## Usage Example

### 1. Upload and Process a Document

Place a document in the `docs/` folder and either:
- Restart the server (auto-processes on startup)
- Or call the refresh endpoint:

```http
POST /api/refresh
```

### 2. View Extracted Writers

```http
GET /api/writers
```

### 3. Get Writers for a Specific Document

```http
GET /api/documents/mydocument.pdf/writers
```

### 4. Search for a Writer

```http
POST /api/writers/search
Content-Type: application/json

{
  "query": "Einstein"
}
```

### 5. View Detailed Writer Information

```http
GET /api/writers/writer_albert_einstein_12345678
```

## Automatic Updates

When you add a new document that mentions an existing writer:

1. The system recognizes the writer (using fuzzy matching)
2. Updates their mention count
3. Adds the new document to their writings list
4. Extracts and merges any new information (achievements, affiliations, etc.)
5. Updates the writings summary

## Configuration

Key configuration in `writer_manager.py`:

- **Similarity Threshold**: 0.85 (85% match required for fuzzy matching)
- **Database Folder**: `writers_db/`
- **Context Snippets Stored**: Up to 20 most recent snippets per writer

## Advanced Features

### Fuzzy Name Matching

The system uses multiple matching strategies:

1. Exact string matching
2. Sequence similarity (SequenceMatcher)
3. Word overlap analysis
4. Substring matching (for name variations)
5. Alias checking

### Information Merging

When updating a writer:

- Achievements are deduplicated and merged
- Affiliations are combined
- Topics are consolidated
- Context snippets are kept (most recent 20)
- Mention counts are accumulated

### Writer Deduplication

Use the merge endpoint to combine duplicate writer entries:

```http
POST /api/writers/merge
```

This is useful when the automatic fuzzy matching doesn't catch all variations.

## Best Practices

1. **Review New Writers**: Periodically review newly extracted writers to ensure accuracy
2. **Add Aliases**: For common name variations, add aliases to improve matching
3. **Merge Duplicates**: If duplicates are found, use the merge endpoint
4. **Check Document Associations**: Verify that writers are correctly associated with their documents

## Troubleshooting

### Writers Not Being Extracted

- Check that the document contains clear writer attribution (e.g., "by Author Name")
- Ensure names are properly capitalized
- Review the extraction patterns in `writer_manager.py`

### Duplicate Writers Created

- The similarity threshold might be too strict (default: 0.85)
- Add aliases to the correct writer entry
- Use the merge endpoint to combine duplicates

### Missing Information

- Information extraction depends on context around writer mentions
- Ensure documents have sufficient biographical information
- More detailed documents will yield better extracted information

## Future Enhancements

Potential future improvements:

- Integration with LLM for deeper biographical analysis
- External data source integration (Wikipedia, academic databases)
- Writer disambiguation (handling writers with same name)
- Citation tracking and analysis
- Co-author network visualization
- Temporal analysis of writer productivity

## Support

For issues or questions about the writer extraction feature, please refer to the main README or open an issue in the repository.
