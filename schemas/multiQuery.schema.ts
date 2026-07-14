import { z } from "zod";

export const MultiQuerySchema = z.object({
  queries: z
    .array(z.string().min(3).max(200).describe("A short alternative search query for vector retrieval"))
    .min(3)
    .max(5)
    .describe("Different versions of the user's question for retrieval"),
});

export type MultiQueryResult = z.infer<typeof MultiQuerySchema>;
