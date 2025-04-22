import { jsPDF } from 'jspdf';
export function generatePDF(summary: string, actionItems: string[], transcript: string) {
  const doc = new jsPDF();
  const lineHeight = 10;
  let yPosition = 20;

  // Add title
  doc.setFontSize(20);
  doc.text('Meeting Summary', 20, yPosition);
  yPosition += lineHeight * 2;

  // Add summary
  doc.setFontSize(12);
  doc.text('Summary:', 20, yPosition);
  yPosition += lineHeight;
  const summaryLines = doc.splitTextToSize(summary, 170);
  doc.text(summaryLines, 20, yPosition);
  yPosition += lineHeight * (summaryLines.length + 1);

  // Add action items
  doc.text('Action Items:', 20, yPosition);
  yPosition += lineHeight;
  actionItems.forEach(item => {
    doc.text(`• ${item}`, 20, yPosition);
    yPosition += lineHeight;
  });

  // Add transcript
  yPosition += lineHeight;
  doc.text('Full Transcript:', 20, yPosition);
  yPosition += lineHeight;
  const transcriptLines = doc.splitTextToSize(transcript, 170);
  doc.text(transcriptLines, 20, yPosition);

  return doc;
}