import { z } from 'zod';

export const ChatResponseSchema = z.object({
  answer: z.string().describe('Chat response'),
  confidence: z.enum(['high', 'medium', 'low']).describe('Confidence in the response'),
  sources: z.array(z.string()).describe('List of sources referenced'),
  keyQuotes: z.array(z.string()).describe('Relevant quotes from sources').default([]),
  followUpQuestions: z.array(z.string()).describe('Suggested follow-up questions').default([]),
});

export type ChatResponse = z.infer<typeof ChatResponseSchema>;
