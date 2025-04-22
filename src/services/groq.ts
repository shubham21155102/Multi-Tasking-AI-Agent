import Groq from 'groq-sdk';

const groq = new Groq({
  apiKey: import.meta.env.VITE_GROQ_API_KEY,
  dangerouslyAllowBrowser:true
});

export async function transcribeAudio(audioData: ArrayBuffer, fileName: string): Promise<{ transcription: string, usedFallback?: boolean }> {
  // Convert ArrayBuffer to a Blob file with proper MIME type
  const fileExtension = fileName.split('.').pop()?.toLowerCase() || 'mp3';
  const mimeType = fileExtension === 'wav' ? 'audio/wav' : 'audio/mpeg';
  
  // Create a blob with the proper MIME type
  const audioBlob = new Blob([audioData], { type: mimeType });
  
  // Create a File object with a name that includes the extension
  const audioFile = new File([audioBlob], `recording.${fileExtension}`, { type: mimeType });
  
  const formData = new FormData();
  formData.append('audio', audioFile);
  
  try {
    // Send to our backend which will use Alibaba Cloud ASR API
    const response = await fetch('http://localhost:8000/api/transcribe', {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const errorData = await response.text();
      console.error('Server error:', errorData);
      throw new Error('Failed to transcribe audio');
    }
    
    const data = await response.json();
    return { 
      transcription: data.transcript,
      usedFallback: data.usedFallback
    };
  } catch (error) {
    console.error('Error transcribing audio:', error);
    throw error;
  }
}

export async function generateSummary(transcript: string) {
  const prompt = `
    Analyze this multilingual meeting transcript and provide:
    1. A concise summary in English
    2. Key action items
    3. Important decisions made

    Transcript:
    ${transcript}
  `;

  const completion = await groq.chat.completions.create({
    messages: [
      {
        role: 'system',
        content: 'You are a multilingual meeting assistant that helps summarize meetings conducted in English, Mandarin, and Cantonese. Provide clear, structured summaries and action items.',
      },
      {
        role: 'user',
        content: prompt,
      },
    ],
    
    model: 'qwen-qwq-32b',
    temperature: 0.3,
    max_tokens: 1024,
  });

  const response = completion.choices[0]?.message?.content || '';
  
  // Parse the response into structured data
  const sections = response.split('\n\n');
  return {
    summary: sections[0]?.replace('Summary:', '').trim() || '',
    actionItems: sections[1]?.split('\n')
      .filter(line => line.startsWith('-') || line.startsWith('•'))
      .map(item => item.replace(/^[-•]\s*/, '')) || [],
  };
}