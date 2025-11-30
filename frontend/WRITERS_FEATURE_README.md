# Writers Feature - Documentation

## Overview

The Writers feature has been integrated into the ChatWithBackend component, allowing users to:
- View writers extracted from documents
- Search for writers in the current document or across all documents
- View detailed information about each writer including their mentions and works

## Features Implemented

### 1. **Writers Mode**
A new mode has been added to the interface alongside Chat, Search, and Analyze modes.

### 2. **Search Modes**
- **Current Document**: Shows writers only from the currently selected document (default)
- **All Documents**: Shows writers from all processed documents in the system

### 3. **Writer Display**
Each writer card shows:
- Writer's name with an avatar (first letter)
- Aliases (also known as)
- Brief description
- Number of documents they appear in
- Total mentions across all documents

### 4. **Writer Details View**
When clicking on a writer, you'll see:
- Full name and all aliases
- Complete description/biography
- Statistics (total mentions, number of documents)
- List of all writings/mentions organized by document
- Preview of mentions from each document
- First seen and last updated timestamps

## API Endpoints Used

The component integrates with the following backend endpoints:

### GET `/api/writers`
Returns all writers in the database.

**Response:**
```json
{
  "writers": [...],
  "total_count": 10,
  "timestamp": "2025-11-30T..."
}
```

### GET `/api/documents/{filename}/writers`
Returns writers from a specific document.

**Response:**
```json
{
  "document": "document.pdf",
  "writers": [...],
  "total_writers": 5,
  "timestamp": "2025-11-30T..."
}
```

### POST `/api/writers/search`
Searches for writers by name or information.

**Request:**
```json
{
  "query": "shakespeare"
}
```

**Response:**
```json
{
  "query": "shakespeare",
  "results": [...],
  "total_results": 2,
  "timestamp": "2025-11-30T..."
}
```

### GET `/api/writers/{writer_id}`
Gets detailed information about a specific writer.

**Response:**
```json
{
  "writer": {
    "id": "writer_id",
    "name": "William Shakespeare",
    "aliases": ["The Bard"],
    "description": "...",
    "total_mentions": 15,
    "writings": {
      "hamlet.pdf": ["mention1", "mention2"],
      ...
    },
    "created_at": "2025-11-30T...",
    "updated_at": "2025-11-30T..."
  },
  "timestamp": "2025-11-30T..."
}
```

## Zustand Store Updates

New state properties:
- `writers`: Array of writer objects
- `selectedWriter`: Currently selected writer for detail view
- `writerSearchMode`: 'current' or 'all'

New methods:
- `fetchAllWriters()`: Fetch all writers
- `fetchDocumentWriters(filename)`: Fetch writers from a specific document
- `searchWriters(query, filename?)`: Search for writers
- `getWriterDetails(writerId)`: Get detailed writer information
- `setWriterSearchMode(mode)`: Set search mode ('current' or 'all')
- `clearSelectedWriter()`: Clear the selected writer
- `clearWriters()`: Clear the writers list

## Component Structure

### Mode States
1. **chat**: Traditional chat interface
2. **search**: Document search
3. **writers**: Writer exploration (NEW)
4. **analyze**: Document analysis

### Writers Mode Views

#### List View
- Shows all writers based on current search mode
- Search input to filter writers
- Click on writer card to view details

#### Detail View
- Comprehensive writer information
- Back button to return to list
- Organized sections for different information types

## Usage Flow

1. **View Default Writers**
   - Select a document
   - Click "Writers" mode
   - See writers from the current document (default mode)

2. **Search All Writers**
   - Click "Writers" mode
   - Change dropdown to "All Documents"
   - Browse or search all writers in the system

3. **Search for Specific Writer**
   - In Writers mode
   - Type writer name in search box
   - Press Enter or click Search
   - Results update in real-time

4. **View Writer Details**
   - Click on any writer card
   - View comprehensive information
   - Click "← Back to Writers" to return

## Configuration

Update the API base URL in `zustandStore.js`:

```javascript
const API_BASE_URL = 'http://localhost:8000'; // Change to your backend URL
```

## Styling

The component uses:
- Tailwind CSS for styling
- Lucide React icons
- Gradient backgrounds for visual hierarchy
- Responsive design patterns

## Icons Used

- `Users`: Writers mode button
- `User`: Individual writer avatar fallback
- `BookOpen`: Writings section
- `FileText`: Document references
- `Loader2`: Loading states

## Error Handling

- Network errors are caught and logged to console
- Failed API calls maintain UI state
- Loading states prevent multiple simultaneous requests

## Future Enhancements

Potential improvements:
1. Pagination for large writer lists
2. Advanced filtering (by document count, mention frequency)
3. Export writer information
4. Writer comparison view
5. Timeline view of writer mentions
6. Integration with document viewer to highlight mentions

## Dependencies

Required packages:
- `react`: ^18.0.0
- `zustand`: ^4.0.0
- `lucide-react`: ^0.263.0
- `tailwindcss`: ^3.0.0

## Installation

1. Copy `zustandStore.js` to your `src` folder
2. Copy `ChatWithBackend.js` to your `src` folder
3. Import and use the component:

```javascript
import ChatWithBackend from './ChatWithBackend';

function App() {
  return (
    <div className="App">
      <ChatWithBackend />
    </div>
  );
}
```

## Troubleshooting

### Writers not loading
- Check API endpoint is accessible
- Verify CORS settings on backend
- Check browser console for errors

### Search not working
- Ensure backend writer extraction has been run
- Verify documents have been processed
- Check API_BASE_URL is correct

### Details not showing
- Verify writer_id format matches backend
- Check API response structure
- Ensure selectedWriter state is updated
