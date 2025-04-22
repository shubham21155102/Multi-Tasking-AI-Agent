import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileAudio, FileText, Search, Download, Loader2, AlertCircle } from 'lucide-react';
import { transcribeAudio, generateSummary } from './services/groq.ts';
import { generatePDF } from './services/pdf';

interface TranscriptionResult {
  text: string;
  summary: string;
  actionItems: string[];
  usedFallback?: boolean;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const audioFile = acceptedFiles[0];
    setFile(audioFile);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.m4a']
    },
    maxFiles: 1
  });

  const processAudio = async () => {
    if (!file) return;
    
    setIsProcessing(true);
    try {
      // Convert file to ArrayBuffer
      const buffer = await file.arrayBuffer();
      
      // Get transcription - pass the file name to the transcription function
      const { transcription, usedFallback } = await transcribeAudio(buffer, file.name);
      
      // Generate summary using Groq
      const { summary, actionItems } = await generateSummary(transcription);
      
      setResult({
        text: transcription,
        summary,
        actionItems,
        usedFallback
      });
    } catch (error) {
      console.error('Error processing audio:', error);
      // Here you would normally show an error message to the user
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadPDF = () => {
    if (!result) return;
    
    const doc = generatePDF(result.summary, result.actionItems, result.text);
    doc.save('meeting-summary.pdf');
  };

  const highlightSearchResults = (text: string) => {
    if (!searchQuery) return text;
    const regex = new RegExp(`(${searchQuery})`, 'gi');
    return text.split(regex).map((part, i) => 
      regex.test(part) ? <mark key={i} className="bg-yellow-200">{part}</mark> : part
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto p-6">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Multilingual Meeting Assistant</h1>
          <p className="text-gray-600">Upload your meeting recording for instant transcription and summary</p>
        </header>

        <div className="space-y-8">
          {/* Upload Section */}
          <div 
            {...getRootProps()} 
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
              ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}`}
          >
            <input {...getInputProps()} />
            <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
            <p className="text-gray-600">
              {isDragActive
                ? "Drop the audio file here"
                : "Drag & drop an audio file, or click to select"}
            </p>
            <p className="text-sm text-gray-500 mt-2">Supports MP3, WAV, M4A</p>
          </div>

          {/* File Info */}
          {file && (
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <div className="flex items-center gap-4">
                <FileAudio className="h-8 w-8 text-blue-500" />
                <div>
                  <h3 className="font-medium">{file.name}</h3>
                  <p className="text-sm text-gray-500">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </div>
                <button
                  onClick={processAudio}
                  disabled={isProcessing}
                  className="ml-auto bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 
                    disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <FileText className="h-4 w-4" />
                      Process Audio
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Results Section */}
          {result && (
            <div className="space-y-6">
              {/* Fallback Notice */}
              {result.usedFallback && (
                <div className="bg-amber-50 border-l-4 border-amber-400 p-4 rounded-md">
                  <div className="flex items-start">
                    <AlertCircle className="h-5 w-5 text-amber-400 mt-0.5 mr-2" />
                    <div>
                      <h3 className="font-medium text-amber-800">Transcription Service Notice</h3>
                      <p className="text-sm text-amber-700 mt-1">
                        The Alibaba Cloud ASR service is currently unavailable. We've provided a placeholder transcription.
                        Please try again later or contact support if this issue persists.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Search Bar */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search transcription..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              {/* Summary Card */}
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <h2 className="text-xl font-semibold mb-4">Meeting Summary</h2>
                <p className="text-gray-700 mb-4">{result.summary}</p>
                
                <h3 className="font-semibold mb-2">Action Items:</h3>
                <ul className="list-disc list-inside space-y-1">
                  {result.actionItems.map((item, index) => (
                    <li key={index} className="text-gray-700">{item}</li>
                  ))}
                </ul>
              </div>

              {/* Transcription Card */}
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-semibold">Full Transcription</h2>
                  <button 
                    onClick={downloadPDF}
                    className="text-blue-500 hover:text-blue-600 flex items-center gap-2"
                  >
                    <Download className="h-4 w-4" />
                    Download PDF
                  </button>
                </div>
                <div className="prose max-w-none">
                  <p className="text-gray-700 whitespace-pre-wrap">{highlightSearchResults(result.text)}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;