import { create } from 'zustand';

const API_BASE_URL = 'http://localhost:8000'; // Adjust based on your backend URL

const useAppStore = create((set, get) => ({
  // Existing state
  currentFile: null,
  messages: [],
  isLoading: false,
  unifiedIndexReady: false,

  // Writer-specific state
  writers: [],
  selectedWriter: null,
  writerSearchMode: 'current', // 'current' or 'all'

  // Set current file
  setCurrentFile: (file) => set({ currentFile: file }),

  // Message management
  addMessage: (message) => set((state) => ({
    messages: [...state.messages, message]
  })),

  clearMessages: () => set({ messages: [] }),

  // Chat with specific document
  sendChatMessage: async (query, filename) => {
    set({ isLoading: true });

    try {
      // Add user message
      get().addMessage({
        type: 'user',
        content: query,
        timestamp: new Date().toISOString()
      });

      const response = await fetch(`${API_BASE_URL}/api/chat/${encodeURIComponent(filename)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });

      if (!response.ok) throw new Error('Chat request failed');

      const data = await response.json();

      // Add assistant response
      get().addMessage({
        type: 'assistant',
        content: data.response,
        sources: data.sources_used,
        timestamp: data.timestamp
      });

      return data;
    } catch (error) {
      get().addMessage({
        type: 'error',
        content: `Error: ${error.message}`,
        timestamp: new Date().toISOString()
      });
      throw error;
    } finally {
      set({ isLoading: false });
    }
  },

  // Analyze document
  analyzeDocument: async (filename) => {
    set({ isLoading: true });

    try {
      const response = await fetch(`${API_BASE_URL}/api/analyze/${encodeURIComponent(filename)}`, {
        method: 'POST'
      });

      if (!response.ok) throw new Error('Analysis request failed');

      return await response.json();
    } catch (error) {
      throw error;
    } finally {
      set({ isLoading: false });
    }
  },

  // Search documents
  searchDocuments: async (query, filename = null, mode = 'unified', top_k = 5) => {
    set({ isLoading: true });

    try {
      const response = await fetch(`${API_BASE_URL}/api/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          filename,
          mode,
          top_k
        })
      });

      if (!response.ok) throw new Error('Search request failed');

      return await response.json();
    } catch (error) {
      throw error;
    } finally {
      set({ isLoading: false });
    }
  },

  // Query documents (unified)
  queryDocuments: async (query) => {
    set({ isLoading: true });

    try {
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });

      if (!response.ok) throw new Error('Query request failed');

      return await response.json();
    } catch (error) {
      throw error;
    } finally {
      set({ isLoading: false });
    }
  },

  // ============================================================================
  // Writer Management Functions
  // ============================================================================

  // Fetch all writers
  fetchAllWriters: async () => {
    set({ isLoading: true });

    try {
      const response = await fetch(`${API_BASE_URL}/api/writers`);

      if (!response.ok) throw new Error('Failed to fetch writers');

      const data = await response.json();
      set({ writers: data.writers });

      return data;
    } catch (error) {
      console.error('Error fetching writers:', error);
      throw error;
    } finally {
      set({ isLoading: false });
    }
  },

  // Fetch writers for current document
  fetchDocumentWriters: async (filename) => {
    set({ isLoading: true });

    try {
      const response = await fetch(`${API_BASE_URL}/api/documents/${encodeURIComponent(filename)}/writers`);

      if (!response.ok) throw new Error('Failed to fetch document writers');

      const data = await response.json();
      set({ writers: data.writers });

      return data;
    } catch (error) {
      console.error('Error fetching document writers:', error);
      throw error;
    } finally {
      set({ isLoading: false });
    }
  },

  // Search for writers
  searchWriters: async (query, filename = null) => {
    set({ isLoading: true });

    try {
      // If searching in current document, filter by document
      if (filename) {
        const docWritersResponse = await fetch(
          `${API_BASE_URL}/api/documents/${encodeURIComponent(filename)}/writers`
        );

        if (!docWritersResponse.ok) throw new Error('Failed to search writers in document');

        const docData = await docWritersResponse.json();

        // Filter writers by query
        const filteredWriters = docData.writers.filter(writer =>
          writer.name.toLowerCase().includes(query.toLowerCase()) ||
          writer.aliases?.some(alias => alias.toLowerCase().includes(query.toLowerCase()))
        );

        set({ writers: filteredWriters });

        return {
          query,
          results: filteredWriters,
          total_results: filteredWriters.length,
          timestamp: new Date().toISOString()
        };
      } else {
        // Search all writers
        const response = await fetch(`${API_BASE_URL}/api/writers/search`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query })
        });

        if (!response.ok) throw new Error('Failed to search writers');

        const data = await response.json();
        set({ writers: data.results });

        return data;
      }
    } catch (error) {
      console.error('Error searching writers:', error);
      throw error;
    } finally {
      set({ isLoading: false });
    }
  },

  // Get writer details
  getWriterDetails: async (writerId) => {
    set({ isLoading: true });

    try {
      const response = await fetch(`${API_BASE_URL}/api/writers/${encodeURIComponent(writerId)}`);

      if (!response.ok) throw new Error('Failed to fetch writer details');

      const data = await response.json();
      set({ selectedWriter: data.writer });

      return data;
    } catch (error) {
      console.error('Error fetching writer details:', error);
      throw error;
    } finally {
      set({ isLoading: false });
    }
  },

  // Set writer search mode
  setWriterSearchMode: (mode) => set({ writerSearchMode: mode }),

  // Clear selected writer
  clearSelectedWriter: () => set({ selectedWriter: null }),

  // Clear writers list
  clearWriters: () => set({ writers: [] })
}));

export default useAppStore;
