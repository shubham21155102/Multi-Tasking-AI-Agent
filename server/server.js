import express from "express";
import multer from "multer";
import path from "path";
import fs from "fs";
import dotenv from "dotenv";
import Groq from "groq-sdk";
import cors from "cors";
import { fileURLToPath } from "url";
import crypto from "crypto";
import axios from "axios";
import { exec } from "child_process";
import { promisify } from "util";

// Remove the LangChain imports that are causing issues
// We'll implement a simpler approach without these specific dependencies

const execPromise = promisify(exec);

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

// Create directories if they don't exist
const uploadsDir = path.join(__dirname, "../uploads");
const chunksDir = path.join(__dirname, "./chunks");
[uploadsDir, chunksDir].forEach(dir => {
  if (!fs.existsSync(dir)){
    fs.mkdirSync(dir, { recursive: true });
  }
});

// Alibaba Cloud ASR API configuration
const ALIBABA_ASR_CONFIG = {
  appKey: "PVI9DP7cToUpEhvN",
  token: "1ec68967dee1422daa102e1e037bf145", // Note: This token expires within 24 hours
  apiEndpoint: "https://nls-gateway-cn-shanghai.aliyuncs.com/stream/v1/file",
  sampleRate: "16000",
};

// Keep Groq for translation only
const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
  dangerouslyAllowBrowser: true,
});

// Audio processing configuration
const AUDIO_CONFIG = {
  chunkDuration: 60, // in seconds
  maxFileSizeForDirectProcessing: 5 * 1024 * 1024, // 5MB
};

// Set up multer storage
const storage = multer.diskStorage({
  destination: path.join(__dirname, "../uploads"),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    const uniqueName = crypto.randomBytes(16).toString("hex") + ext;
    cb(null, uniqueName);
  },
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    // Extract extension from the original filename
    const originalExt = path.extname(file.originalname).toLowerCase();
    
    // Check mime type as a backup if extension is missing
    let fileExtension = originalExt;
    if (!fileExtension || fileExtension === '.') {
      if (file.mimetype === 'audio/mpeg' || file.mimetype === 'audio/mp3') {
        fileExtension = '.mp3';
      } else if (file.mimetype === 'audio/wav' || file.mimetype === 'audio/x-wav') {
        fileExtension = '.wav';
      } else if (file.mimetype === 'audio/m4a' || file.mimetype === 'audio/x-m4a') {
        fileExtension = '.m4a';
      }
    }
    
    // Log for debugging
    console.log("MIME type:", file.mimetype);
    console.log("Original file extension:", originalExt);
    console.log("Determined file extension:", fileExtension);
    
    // Check against allowed extensions
    const allowedExt = ['.mp3', '.wav', '.m4a'];
    if (allowedExt.includes(fileExtension)) {
      // If original extension is missing, we can add it to the file
      if (!originalExt || originalExt === '.') {
        file.originalname = `${file.originalname}${fileExtension}`;
      }
      cb(null, true);
    } else {
      cb(new Error(`Unsupported file extension: ${fileExtension}. Only .mp3, .wav, and .m4a are supported.`));
    }
  }
});

// Function to use ffmpeg to trim/chunk an audio file
async function processAudioWithFFmpeg(inputFile, outputPrefix) {
  try {
    const stats = fs.statSync(inputFile);
    const fileSizeInBytes = stats.size;
    
    // For small files, no need to chunk
    if (fileSizeInBytes <= AUDIO_CONFIG.maxFileSizeForDirectProcessing) {
      console.log("File is small enough for direct processing");
      return [inputFile];
    }
    
    console.log(`Processing large audio file (${fileSizeInBytes / (1024 * 1024)} MB)`);
    
    // Get audio duration
    const { stdout: durationInfo } = await execPromise(
      `ffmpeg -i "${inputFile}" 2>&1 | grep "Duration"`
    );
    
    // Parse duration
    const durationMatch = durationInfo.match(/Duration: (\d{2}):(\d{2}):(\d{2})/);
    if (!durationMatch) {
      throw new Error("Could not determine audio duration");
    }
    
    const hours = parseInt(durationMatch[1]);
    const minutes = parseInt(durationMatch[2]);
    const seconds = parseInt(durationMatch[3]);
    const totalSeconds = hours * 3600 + minutes * 60 + seconds;
    
    console.log(`Audio duration: ${totalSeconds} seconds`);
    
    // Calculate number of chunks
    const numChunks = Math.ceil(totalSeconds / AUDIO_CONFIG.chunkDuration);
    const outputFiles = [];
    
    // Create chunks
    for (let i = 0; i < numChunks; i++) {
      const startTime = i * AUDIO_CONFIG.chunkDuration;
      const outputFile = `${outputPrefix}_chunk${i}${path.extname(inputFile)}`;
      
      console.log(`Creating chunk ${i+1}/${numChunks}: ${startTime}s to ${startTime + AUDIO_CONFIG.chunkDuration}s`);
      
      const command = `ffmpeg -ss ${startTime} -t ${AUDIO_CONFIG.chunkDuration} -i "${inputFile}" -acodec copy "${outputFile}" -y`;
      await execPromise(command);
      
      outputFiles.push(outputFile);
    }
    
    return outputFiles;
  } catch (error) {
    console.error("Error processing audio with ffmpeg:", error);
    throw error;
  }
}

// Function to transcribe chunks and combine them
async function transcribeAudioChunks(chunks) {
  const transcriptions = [];
  
  for (let i = 0; i < chunks.length; i++) {
    console.log(`Transcribing chunk ${i+1}/${chunks.length}`);
    
    // Try primary method (Alibaba ASR)
    let transcription = await transcribeWithAlibabaASR(chunks[i]);
    
    // If primary method fails, transcribeWithAlibabaASR will already try fallbacks
    transcriptions.push(transcription);
  }
  
  // Combine transcriptions
  return transcriptions.join(" ");
}

// Main function to process audio file
async function processAudioFile(filePath) {
  try {
    const filename = path.basename(filePath, path.extname(filePath));
    const chunkOutputPrefix = path.join(chunksDir, filename);
    
    // Process audio file with ffmpeg to get chunks
    const chunks = await processAudioWithFFmpeg(filePath, chunkOutputPrefix);
    
    // Transcribe all chunks
    const transcription = await transcribeAudioChunks(chunks);
    
    // Clean up chunks
    chunks.forEach(chunk => {
      if (chunk !== filePath) { // Don't delete the original file
        fs.unlinkSync(chunk);
      }
    });
    
    return transcription;
  } catch (error) {
    console.error("Error processing audio file:", error);
    throw error;
  }
}

// Function to transcribe audio using Alibaba Cloud ASR
async function transcribeWithAlibabaASR(filePath) {
  // Get file extension
  const ext = path.extname(filePath).toLowerCase().substring(1); // Remove dot from extension
  
  // Construct the URL with query parameters
  const url = `${ALIBABA_ASR_CONFIG.apiEndpoint}?appkey=${ALIBABA_ASR_CONFIG.appKey}&token=${ALIBABA_ASR_CONFIG.token}&format=${ext}&sample_rate=${ALIBABA_ASR_CONFIG.sampleRate}`;
  
  // Read file as binary data
  const fileData = fs.readFileSync(filePath);
  
  // Define retry strategy
  const MAX_RETRIES = 2;
  let retries = 0;
  
  while (retries <= MAX_RETRIES) {
    try {
      console.log(`Attempt ${retries + 1}/${MAX_RETRIES + 1} to call Alibaba ASR API...`);
      
      // Configure timeout for the request (10 seconds)
      const response = await axios.post(url, fileData, {
        headers: {
          'Content-Type': 'application/octet-stream'
        },
        timeout: 10000 // 10 second timeout
      });
      
      // Check for API errors
      if (response.data.status !== 20000000) {
        throw new Error(`Alibaba ASR API error: ${response.data.message}`);
      }
      
      console.log("Transcription successful");
      return response.data.result;
    } catch (error) {
      retries++;
      console.error(`Alibaba ASR transcription error (attempt ${retries}/${MAX_RETRIES + 1}):`, error.message);
      
      if (retries <= MAX_RETRIES) {
        // Wait before retrying (exponential backoff)
        const delay = 1000 * (2 ** retries);
        console.log(`Retrying in ${delay/1000} seconds...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      } else {
        // All retries failed, try Groq's transcription service
        console.error("All retry attempts failed. Using Groq's transcription service as fallback.");
        return await transcribeWithGroq(filePath);
      }
    }
  }
}

// Fallback transcription using Groq
async function transcribeWithGroq(filePath) {
  try {
    console.log("Attempting transcription with Groq's service...");
    const transcription = await groq.audio.transcriptions.create({
      file: fs.createReadStream(filePath),
      model: "whisper-large-v3-turbo",
      response_format: "verbose_json",
    });
    
    console.log("Groq transcription successful");
    return transcription.text;
  } catch (error) {
    console.error("Groq transcription error:", error.message);
    // If even Groq fails, return mock transcription
    return getMockTranscription(filePath);
  }
}

// Mock transcription function as last resort fallback
function getMockTranscription(filePath) {
  const filename = path.basename(filePath);
  console.log(`Using mock transcription for file: ${filename}`);
  
  return `[This is a mock transcription because both Alibaba Cloud ASR and Groq transcription services are currently unavailable. The actual transcription would appear here.]`;
}

// Function to improve transcription quality using Groq directly instead of LangChain
async function enhanceTranscriptionWithGroq(rawTranscription) {
  try {
    console.log("Enhancing transcription with Groq...");
    
    // If the transcription is too short, no need for enhancement
    if (rawTranscription.length < 1000) {
      return rawTranscription;
    }
    
    // Create a simple function to split text into manageable chunks
    const splitTextIntoChunks = (text, chunkSize = 4000, overlap = 200) => {
      const chunks = [];
      let startIndex = 0;
      
      while (startIndex < text.length) {
        // Calculate end index for this chunk
        let endIndex = Math.min(startIndex + chunkSize, text.length);
        
        // If we're not at the end of the text and this isn't the first chunk,
        // try to find a good break point (period, question mark, etc.)
        if (endIndex < text.length && endIndex !== text.length) {
          // Look for a period, question mark, or exclamation point followed by a space or newline
          const breakPointRegex = /[.!?]\s+/g;
          let lastBreakPoint = startIndex;
          let match;
          
          // Find the last good break point in this chunk
          while ((match = breakPointRegex.exec(text.substring(startIndex, endIndex + 10))) !== null) {
            lastBreakPoint = startIndex + match.index + 2; // +2 to include the punctuation and space
          }
          
          // Use the break point if we found one
          if (lastBreakPoint > startIndex) {
            endIndex = lastBreakPoint;
          }
        }
        
        // Add this chunk to our list
        chunks.push(text.substring(startIndex, endIndex));
        
        // Move to the next chunk, with overlap
        startIndex = endIndex - overlap;
      }
      
      return chunks;
    };
    
    // Split the text into manageable chunks
    const textChunks = splitTextIntoChunks(rawTranscription);
    console.log(`Split transcription into ${textChunks.length} chunks for enhancement`);
    
    // Process each chunk to improve/correct transcription
    const enhancedChunks = [];
    for (let i = 0; i < textChunks.length; i++) {
      const chunk = textChunks[i];
      try {
        console.log(`Enhancing chunk ${i+1}/${textChunks.length}`);
        
        const prompt = `You are a transcription correction assistant. Your job is to fix any errors in this automatic speech recognition output. 
        Make minimal changes, only correcting obvious errors while preserving the original speaker's words and intent.
        
        Original transcription:
        ${chunk}
        
        Enhanced transcription:`;
        
        // Use Groq for enhancement
        const completion = await groq.chat.completions.create({
          messages: [
            {
              role: "user",
              content: prompt,
            },
          ],
          model: "qwen-qwq-32b",
          temperature: 0.2, // Lower temperature for more accurate corrections
          max_completion_tokens: 4096,
          top_p: 0.95,
        });
        
        const enhancedText = completion.choices[0]?.message?.content || chunk;
        enhancedChunks.push(enhancedText);
      } catch (error) {
        console.error(`Error enhancing chunk ${i+1}:`, error.message);
        enhancedChunks.push(chunk); // Fall back to original chunk on error
      }
    }
    
    // Join the enhanced chunks
    return enhancedChunks.join(" ");
  } catch (error) {
    console.error("Error enhancing transcription:", error);
    return rawTranscription; // Return original transcription if enhancement fails
  }
}

app.post("/api/transcribe", upload.single("audio"), async (req, res) => {
  try {
    const filePath = path.join(__dirname, "../uploads", req.file.filename);
    console.log("Processing file:", filePath);

    // Process audio file (chunking and transcription)
    let transcriptionText = await processAudioFile(filePath);
    
    // If transcription failed, use fallback
    let usedFallback = false;
    if (!transcriptionText) {
      usedFallback = true;
      transcriptionText = getMockTranscription(filePath);
    }

    // Enhance transcription with Groq
    const enhancedTranscription = await enhanceTranscriptionWithGroq(transcriptionText);

    // Translate the text with Groq
    const prompt = `Translate to English:\n${enhancedTranscription}`;

    const chatCompletion = await groq.chat.completions.create({
      messages: [
        {
          role: "user",
          content: prompt,
        },
      ],
      model: "qwen-qwq-32b",
      temperature: 0.6,
      max_completion_tokens: 4096,
      top_p: 0.95,
    });

    const translated = chatCompletion.choices[0]?.message?.content || "Translation failed";

    // Add a flag to indicate whether fallback was used
    res.json({ 
      transcript: enhancedTranscription, 
      translated,
      usedFallback
    });

    // Optional: Clean up uploaded file
    fs.unlink(filePath, (err) => {
      if (err) console.error("Failed to delete uploaded file:", err);
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ 
      error: err.message || "Server Error",
      details: err.toString()
    });
  }
});

app.listen(8000, () => {
  console.log("🚀 Server running on http://localhost:8000");
});