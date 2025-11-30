import React, { useState, useRef, useEffect } from 'react';
import useAppStore from './zustandStore';
import { Send, Loader2, Search, FileText, MessageSquare, Sparkles, Users, User, BookOpen } from 'lucide-react';

const ChatWithBackend = () => {
  const {
    currentFile,
    messages,
    isLoading,
    sendChatMessage,
    analyzeDocument,
    searchDocuments,
    queryDocuments,
    clearMessages,
    unifiedIndexReady,
    // Writer-related
    writers,
    selectedWriter,
    writerSearchMode,
    fetchAllWriters,
    fetchDocumentWriters,
    searchWriters,
    getWriterDetails,
    setWriterSearchMode,
    clearSelectedWriter,
    clearWriters
  } = useAppStore();

  const [input, setInput] = useState('');
  const [mode, setMode] = useState('chat'); // 'chat', 'search', 'analyze', 'writers'
  const [searchMode, setSearchMode] = useState('unified'); // 'unified', 'individual', 'hybrid'
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load writers when switching to writer mode
  useEffect(() => {
    if (mode === 'writers') {
      loadWriters();
    } else {
      clearWriters();
      clearSelectedWriter();
    }
  }, [mode, writerSearchMode, currentFile]);

  const loadWriters = async () => {
    try {
      if (writerSearchMode === 'current' && currentFile) {
        await fetchDocumentWriters(currentFile.name);
      } else {
        await fetchAllWriters();
      }
    } catch (error) {
      console.error('Error loading writers:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const query = input.trim();
    setInput('');

    try {
      if (mode === 'chat') {
        if (!currentFile) {
          // Use unified query if no file selected
          const result = await queryDocuments(query);
          // Display result as a message
          useAppStore.getState().addMessage({
            type: 'user',
            content: query,
            timestamp: new Date().toISOString()
          });
          useAppStore.getState().addMessage({
            type: 'assistant',
            content: result.response,
            sources: result.sources_used,
            timestamp: result.timestamp
          });
        } else {
          // Chat with specific document
          await sendChatMessage(query, currentFile.name);
        }
      } else if (mode === 'search') {
        // Perform search
        const result = await searchDocuments(
          query,
          searchMode === 'individual' ? currentFile?.name : null,
          searchMode,
          5
        );

        // Display search results
        useAppStore.getState().addMessage({
          type: 'user',
          content: `Search: ${query}`,
          timestamp: new Date().toISOString()
        });

        console.log(result);

        const resultsMessage = result.results.map((r, i) =>
          `${i + 1}. [${r.source_file}] (Score: ${r.score.toFixed(3)})\n${r.content.substring(0, 200)}...`
        ).join('\n\n');

        useAppStore.getState().addMessage({
          type: 'assistant',
          content: `Found ${result.total_results} results:\n\n${resultsMessage}`,
          timestamp: result.timestamp
        });
      } else if (mode === 'writers') {
        // Search for writers
        const result = await searchWriters(
          query,
          writerSearchMode === 'current' ? currentFile?.name : null
        );

        console.log('Writer search results:', result);
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const handleAnalyze = async () => {
    if (!currentFile || isLoading) return;

    try {
      const result = await analyzeDocument(currentFile.name);

      useAppStore.getState().addMessage({
        type: 'system',
        content: `Analysis requested for: ${currentFile.name}`,
        timestamp: new Date().toISOString()
      });

      useAppStore.getState().addMessage({
        type: 'assistant',
        content: result.analysis,
        timestamp: result.timestamp
      });
    } catch (error) {
      console.error('Analysis error:', error);
      useAppStore.getState().addMessage({
        type: 'error',
        content: `Failed to analyze document: ${error.message}`,
        timestamp: new Date().toISOString()
      });
    }
  };

  const handleWriterClick = async (writer) => {
    try {
      await getWriterDetails(writer.id);
    } catch (error) {
      console.error('Error fetching writer details:', error);
    }
  };

  const handleBackToWriters = () => {
    clearSelectedWriter();
  };

  const getMessageIcon = (type) => {
    switch (type) {
      case 'user':
        return <div className="w-8 h-8 bg-indigo-600 rounded-full flex items-center justify-center text-white text-sm">U</div>;
      case 'assistant':
        return <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center text-white"><Sparkles className="w-4 h-4" /></div>;
      case 'system':
        return <div className="w-8 h-8 bg-gray-600 rounded-full flex items-center justify-center text-white"><MessageSquare className="w-4 h-4" /></div>;
      case 'error':
        return <div className="w-8 h-8 bg-red-600 rounded-full flex items-center justify-center text-white">!</div>;
      default:
        return null;
    }
  };

  const renderWriterDetails = () => {
    if (!selectedWriter) return null;

    return (
      <div className="p-6 space-y-6">
        <button
          onClick={handleBackToWriters}
          className="text-indigo-600 hover:text-indigo-800 flex items-center gap-2 mb-4"
        >
          ← Back to Writers
        </button>

        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg p-6 border border-indigo-200">
          <div className="flex items-start gap-4">
            <div className="w-16 h-16 bg-indigo-600 rounded-full flex items-center justify-center text-white text-2xl font-bold">
              {selectedWriter.name.charAt(0).toUpperCase()}
            </div>
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-gray-800">{selectedWriter.name}</h2>
              {selectedWriter.aliases && selectedWriter.aliases.length > 0 && (
                <p className="text-sm text-gray-600 mt-1">
                  Also known as: {selectedWriter.aliases.join(', ')}
                </p>
              )}
            </div>
          </div>
        </div>

        {selectedWriter.description && (
          <div className="bg-white rounded-lg p-4 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-800 mb-2">About</h3>
            <p className="text-gray-700 whitespace-pre-wrap">{selectedWriter.description}</p>
          </div>
        )}

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white rounded-lg p-4 border border-gray-200">
            <div className="text-sm text-gray-600">Total Mentions</div>
            <div className="text-2xl font-bold text-indigo-600">{selectedWriter.total_mentions || 0}</div>
          </div>
          <div className="bg-white rounded-lg p-4 border border-gray-200">
            <div className="text-sm text-gray-600">Documents</div>
            <div className="text-2xl font-bold text-purple-600">
              {selectedWriter.writings ? Object.keys(selectedWriter.writings).length : 0}
            </div>
          </div>
        </div>

        {selectedWriter.writings && Object.keys(selectedWriter.writings).length > 0 && (
          <div className="bg-white rounded-lg p-4 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center gap-2">
              <BookOpen className="w-5 h-5" />
              Writings & Mentions
            </h3>
            <div className="space-y-3">
              {Object.entries(selectedWriter.writings).map(([docName, mentions]) => (
                <div key={docName} className="border-l-4 border-indigo-300 pl-3 py-2">
                  <div className="flex items-center justify-between">
                    <div className="font-medium text-gray-800">{docName}</div>
                    <div className="text-sm text-gray-600 bg-gray-100 px-2 py-1 rounded">
                      {mentions.length} mention{mentions.length !== 1 ? 's' : ''}
                    </div>
                  </div>
                  {mentions.slice(0, 3).map((mention, idx) => (
                    <div key={idx} className="text-sm text-gray-600 mt-1 pl-3 border-l-2 border-gray-200">
                      {mention.substring(0, 150)}...
                    </div>
                  ))}
                  {mentions.length > 3 && (
                    <div className="text-sm text-indigo-600 mt-1 pl-3">
                      +{mentions.length - 3} more mention{mentions.length - 3 !== 1 ? 's' : ''}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {selectedWriter.created_at && (
          <div className="text-sm text-gray-500 text-center">
            First seen: {new Date(selectedWriter.created_at).toLocaleDateString()}
            {selectedWriter.updated_at && selectedWriter.updated_at !== selectedWriter.created_at && (
              <> • Updated: {new Date(selectedWriter.updated_at).toLocaleDateString()}</>
            )}
          </div>
        )}
      </div>
    );
  };

  const renderWritersList = () => {
    if (isLoading) {
      return (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 animate-spin text-indigo-600" />
          <span className="ml-2 text-gray-600">Loading writers...</span>
        </div>
      );
    }

    if (writers.length === 0) {
      return (
        <div className="text-center py-20 text-gray-500">
          <Users className="w-16 h-16 mx-auto mb-4 opacity-30" />
          <p className="text-lg font-medium mb-2">
            {writerSearchMode === 'current' && currentFile
              ? `No writers found in ${currentFile.name}`
              : 'No writers found'}
          </p>
          <p className="text-sm">
            {writerSearchMode === 'current'
              ? 'Try searching in all documents or upload a document with writer information'
              : 'Upload documents to extract writer information'}
          </p>
        </div>
      );
    }

    return (
      <div className="p-4 space-y-3">
        <div className="text-sm text-gray-600 mb-4">
          Found {writers.length} writer{writers.length !== 1 ? 's' : ''}
          {writerSearchMode === 'current' && currentFile && ` in ${currentFile.name}`}
        </div>
        {writers.map((writer) => (
          <div
            key={writer.id}
            onClick={() => handleWriterClick(writer)}
            className="bg-white border border-gray-200 rounded-lg p-4 hover:border-indigo-400 hover:shadow-md transition-all cursor-pointer"
          >
            <div className="flex items-start gap-3">
              <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-full flex items-center justify-center text-white text-lg font-bold flex-shrink-0">
                {writer.name.charAt(0).toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-gray-800 mb-1">{writer.name}</h3>
                {writer.aliases && writer.aliases.length > 0 && (
                  <p className="text-xs text-gray-500 mb-2">
                    aka: {writer.aliases.slice(0, 2).join(', ')}
                    {writer.aliases.length > 2 && ` +${writer.aliases.length - 2} more`}
                  </p>
                )}
                {writer.description && (
                  <p className="text-sm text-gray-600 line-clamp-2">
                    {writer.description.substring(0, 150)}...
                  </p>
                )}
                <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                  <span className="flex items-center gap-1">
                    <FileText className="w-3 h-3" />
                    {writer.writings ? Object.keys(writer.writings).length : 0} documents
                  </span>
                  <span>
                    {writer.total_mentions || 0} mentions
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="flex-1 bg-white/95 backdrop-blur-lg rounded-2xl shadow-2xl flex flex-col overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xl font-semibold">
            {mode === 'writers'
              ? selectedWriter
                ? selectedWriter.name
                : writerSearchMode === 'current' && currentFile
                  ? `Writers in: ${currentFile.name}`
                  : 'All Writers'
              : currentFile
                ? `Chat with: ${currentFile.name}`
                : 'Universal Document Chat'}
          </h2>
          <div className="flex items-center gap-2">
            {unifiedIndexReady && (
              <span className="text-xs bg-green-500 px-2 py-1 rounded-full">
                Unified Index Ready
              </span>
            )}
            <button
              onClick={mode === 'writers' ? () => { clearWriters(); clearSelectedWriter(); } : clearMessages}
              className="text-xs bg-white/20 hover:bg-white/30 px-3 py-1 rounded-lg transition-colors"
            >
              {mode === 'writers' ? 'Refresh' : 'Clear Chat'}
            </button>
          </div>
        </div>

        {/* Mode Selector */}
        <div className="flex gap-2">
          <button
            onClick={() => setMode('chat')}
            className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
              mode === 'chat'
                ? 'bg-white text-indigo-600'
                : 'bg-white/20 text-white hover:bg-white/30'
            }`}
          >
            <MessageSquare className="w-4 h-4 inline mr-1" />
            Chat
          </button>
          <button
            onClick={() => setMode('search')}
            className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
              mode === 'search'
                ? 'bg-white text-indigo-600'
                : 'bg-white/20 text-white hover:bg-white/30'
            }`}
          >
            <Search className="w-4 h-4 inline mr-1" />
            Search
          </button>
          <button
            onClick={() => setMode('writers')}
            className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
              mode === 'writers'
                ? 'bg-white text-indigo-600'
                : 'bg-white/20 text-white hover:bg-white/30'
            }`}
          >
            <Users className="w-4 h-4 inline mr-1" />
            Writers
          </button>
          <button
            onClick={handleAnalyze}
            disabled={!currentFile}
            className="flex-1 px-3 py-2 rounded-lg text-sm font-medium bg-white/20 text-white hover:bg-white/30 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Sparkles className="w-4 h-4 inline mr-1" />
            Analyze
          </button>
        </div>

        {/* Search Mode Selector (only visible in search mode) */}
        {mode === 'search' && (
          <div className="flex gap-2 mt-2">
            <select
              value={searchMode}
              onChange={(e) => setSearchMode(e.target.value)}
              className="flex-1 px-3 py-1 rounded-lg text-sm bg-white/20 text-white border border-white/30"
            >
              <option value="unified" className="text-gray-800">Unified (All Docs)</option>
              <option value="individual" className="text-gray-800">Individual (Current Doc)</option>
              <option value="hybrid" className="text-gray-800">Hybrid</option>
            </select>
          </div>
        )}

        {/* Writer Search Mode Selector (only visible in writers mode) */}
        {mode === 'writers' && !selectedWriter && (
          <div className="flex gap-2 mt-2">
            <select
              value={writerSearchMode}
              onChange={(e) => setWriterSearchMode(e.target.value)}
              className="flex-1 px-3 py-1 rounded-lg text-sm bg-white/20 text-white border border-white/30"
            >
              <option value="current" className="text-gray-800">
                {currentFile ? `Current Document (${currentFile.name})` : 'Current Document (None Selected)'}
              </option>
              <option value="all" className="text-gray-800">All Documents</option>
            </select>
          </div>
        )}
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto">
        {mode === 'writers' ? (
          selectedWriter ? renderWriterDetails() : renderWritersList()
        ) : (
          <div className="p-4 space-y-4">
            {messages.length === 0 ? (
              <div className="text-center py-20 text-gray-500">
                <MessageSquare className="w-16 h-16 mx-auto mb-4 opacity-30" />
                <p className="text-lg font-medium mb-2">
                  {currentFile ? 'Start chatting with your document' : 'Search across all documents'}
                </p>
                <p className="text-sm">
                  {mode === 'chat'
                    ? 'Ask questions about the content'
                    : 'Search for information across documents'}
                </p>
              </div>
            ) : (
              messages.map((message, index) => (
                <div key={index} className={`flex gap-3 ${
                  message.type === 'user' ? 'justify-end' : 'justify-start'
                }`}>
                  {message.type !== 'user' && getMessageIcon(message.type)}
                  <div className={`max-w-3xl rounded-lg px-4 py-3 ${
                    message.type === 'user'
                      ? 'bg-indigo-600 text-white'
                      : message.type === 'error'
                      ? 'bg-red-50 text-red-800 border border-red-200'
                      : message.type === 'system'
                      ? 'bg-gray-100 text-gray-700 border border-gray-200'
                      : 'bg-gray-50 text-gray-800 border border-gray-200'
                  }`}>
                    <div className="whitespace-pre-wrap">{message.content}</div>
                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-2 pt-2 border-t border-gray-300 text-xs">
                        <span className="font-semibold">Sources:</span>
                        {message.sources.map((source, i) => (
                          <div key={i} className="mt-1">
                            <FileText className="w-3 h-3 inline mr-1" />
                            {source.file}: {source.content_preview.substring(0, 100)}...
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  {message.type === 'user' && getMessageIcon(message.type)}
                </div>
              ))
            )}
            {isLoading && (
              <div className="flex items-center gap-2 text-gray-500">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Processing...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area - Only show for chat, search, and writer search modes */}
      {(mode === 'chat' || mode === 'search' || (mode === 'writers' && !selectedWriter)) && (
        <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                mode === 'chat'
                  ? `Ask about ${currentFile ? currentFile.name : 'all documents'}...`
                  : mode === 'search'
                    ? `Search in ${searchMode === 'unified' ? 'all documents' : currentFile ? currentFile.name : 'documents'}...`
                    : `Search for a writer${writerSearchMode === 'current' && currentFile ? ` in ${currentFile.name}` : ''}...`
              }
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : mode === 'search' || mode === 'writers' ? (
                <Search className="w-5 h-5" />
              ) : (
                <Send className="w-5 h-5" />
              )}
              {mode === 'search' || mode === 'writers' ? 'Search' : 'Send'}
            </button>
          </div>
        </form>
      )}
    </div>
  );
};

export default ChatWithBackend;
