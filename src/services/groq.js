import fs from "fs";
import dotenv from "dotenv";
import { jsPDF } from "jspdf";
import axios from "axios";
import path from "path";
dotenv.config();
import Groq from "groq-sdk";

const groq = new Groq({
  apiKey: "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",
  dangerouslyAllowBrowser: true
});

// Alibaba Cloud ASR API configuration
const ALIBABA_ASR_CONFIG = {
  appKey: "PVI9DP7cToUpEhvN",
  token: "1ec68967dee1422daa102e1e037bf145", // Note: This token expires within 24 hours
  apiEndpoint: "https://nls-gateway-cn-shanghai.aliyuncs.com/stream/v1/file",
  sampleRate: "16000",
};

// Function to transcribe audio using Alibaba Cloud ASR
export async function transcribeWithAlibabaASR(filePath) {
  try {
    // Get file extension
    const ext = path.extname(filePath).toLowerCase().substring(1); // Remove dot from extension
    
    // Construct the URL with query parameters
    const url = `${ALIBABA_ASR_CONFIG.apiEndpoint}?appkey=${ALIBABA_ASR_CONFIG.appKey}&token=${ALIBABA_ASR_CONFIG.token}&format=${ext}&sample_rate=${ALIBABA_ASR_CONFIG.sampleRate}`;
    
    // Read file as binary data
    const fileData = fs.readFileSync(filePath);
    
    // Make the API request
    const response = await axios.post(url, fileData, {
      headers: {
        'Content-Type': 'application/octet-stream'
      }
    });
    
    // Check for API errors
    if (response.data.status !== 20000000) {
      throw new Error(`Alibaba ASR API error: ${response.data.message}`);
    }
    
    return response.data.result;
  } catch (error) {
    console.error('Alibaba ASR transcription error:', error);
    throw error;
  }
}

// Translate the transcript to English
export async function translateText(prompt) {
  let translated = "";
  const chatCompletion = await groq.chat.completions.create({
    messages: [
      {
        role: "user",
        content: `Translate the following to English:\n${prompt}`
      }
    ],
    model: "meta-llama/llama-4-scout-17b-16e-instruct",
    temperature: 0.6,
    max_completion_tokens: 4096,
    top_p: 0.95,
    stream: true,
    stop: null
  });

  for await (const chunk of chatCompletion) {
    const text = chunk.choices[0]?.delta?.content || '';
    process.stdout.write(text);
    translated += text;
  }

  return translated;
}

// Ask Groq to summarize and extract action items
export async function summarizeAndExtractActions(translatedText) {
  const chatCompletion = await groq.chat.completions.create({
    messages: [
      {
        role: "user",
        content: `Here is a translated meeting transcript. Summarize it briefly and list clear action items separately:\n\n${translatedText}`
      }
    ],
    model: "qwen-qwq-32b",
    temperature: 0.5,
    max_completion_tokens: 2048
  });

  const reply = chatCompletion.choices[0]?.message?.content || "";

  // Very basic parsing – you can make this more robust
  const summaryMatch = reply.match(/Summary:\s*([\s\S]*?)(?:\n\n|Action Items:)/i);
  const actionsMatch = reply.match(/Action Items:\s*([\s\S]*)/i);

  const summary = summaryMatch ? summaryMatch[1].trim() : "Summary not found.";
  const actionItems = actionsMatch
    ? actionsMatch[1].split('\n').map(item => item.replace(/^[-•\d.]\s*/, '').trim()).filter(Boolean)
    : [];

  return { summary, actionItems };
}

export function generatePDF(summary, actionItems, transcript) {
  const doc = new jsPDF();
  const lineHeight = 10;
  let yPosition = 20;

  doc.setFontSize(20);
  doc.text('Meeting Summary', 20, yPosition);
  yPosition += lineHeight * 2;

  doc.setFontSize(12);
  doc.text('Summary:', 20, yPosition);
  yPosition += lineHeight;
  const summaryLines = doc.splitTextToSize(summary, 170);
  doc.text(summaryLines, 20, yPosition);
  yPosition += lineHeight * (summaryLines.length + 1);

  doc.text('Action Items:', 20, yPosition);
  yPosition += lineHeight;
  actionItems.forEach(item => {
    doc.text(`• ${item}`, 20, yPosition);
    yPosition += lineHeight;
  });

  yPosition += lineHeight;
  doc.text('Full Transcript:', 20, yPosition);
  yPosition += lineHeight;
  const transcriptLines = doc.splitTextToSize(transcript, 170);
  doc.text(transcriptLines, 20, yPosition);

  return doc;
}

// Keep the main function for direct execution
async function main() {
  // Use Alibaba Cloud ASR instead of Groq for transcription
  const audioFilePath = "/Users/shubham/Downloads/AgentSphere/Multilingual-Note-Taking-Agent/src/services/audio/output.m4a";
  console.log("Transcribing audio file:", audioFilePath);
  
  const transcriptionText = await transcribeWithAlibabaASR(audioFilePath);
  console.log("Transcription complete. Now translating...");

  const translatedText = await translateText(transcriptionText);
  const { summary, actionItems } = await summarizeAndExtractActions(translatedText);
  const pdf = generatePDF(summary, actionItems, translatedText);
  
  // Save PDF to disk
  pdf.save("Meeting_Report.pdf");
  console.log("\nPDF saved as 'Meeting_Report.pdf'");
}

// Only run main if this file is executed directly (not imported)
if (import.meta.url === import.meta.main) {
  main();
}