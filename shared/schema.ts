import { pgTable, text, serial, integer, boolean, timestamp, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const datasets = pgTable("datasets", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  source: text("source"),
  description: text("description"),
  url: text("url"),
  format: text("format"),
  size: text("size"),
  tags: text("tags").array(),
  dateAdded: timestamp("date_added").defaultNow().notNull(),
  qualityScore: integer("quality_score"), // 0-100 score
  isProcessed: boolean("is_processed").default(false),
  isFairCompliant: boolean("is_fair_compliant").default(false),
  fairScore: integer("fair_score"), // 0-100 score
  schemaOrgScore: integer("schema_org_score"), // 0-100 score
  status: text("status").default("queued"), // queued, processing, processed, error
  metadata: jsonb("metadata"), // structured metadata
  userId: integer("user_id").references(() => users.id),
});

export const processingQueue = pgTable("processing_queue", {
  id: serial("id").primaryKey(),
  datasetId: integer("dataset_id").references(() => datasets.id).notNull(),
  progress: integer("progress").default(0), // 0-100
  status: text("status").notNull(), // queued, processing, completed, error
  startTime: timestamp("start_time"),
  endTime: timestamp("end_time"),
  estimatedCompletionTime: timestamp("estimated_completion_time"),
  error: text("error"),
});

export const metadataQuality = pgTable("metadata_quality", {
  id: serial("id").primaryKey(),
  datasetId: integer("dataset_id").references(() => datasets.id).notNull(),
  fairFindable: integer("fair_findable"), // 0-100
  fairAccessible: integer("fair_accessible"), // 0-100
  fairInteroperable: integer("fair_interoperable"), // 0-100
  fairReusable: integer("fair_reusable"), // 0-100
  schemaOrgRequired: integer("schema_org_required"), // 0-100
  schemaOrgRecommended: integer("schema_org_recommended"), // 0-100
  schemaOrgVocabulary: integer("schema_org_vocabulary"), // 0-100
  schemaOrgStructure: integer("schema_org_structure"), // 0-100
});

// Insert schemas
export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export const insertDatasetSchema = createInsertSchema(datasets).omit({
  id: true,
  dateAdded: true,
  isProcessed: true,
  isFairCompliant: true,
  fairScore: true,
  schemaOrgScore: true,
});

export const insertProcessingQueueSchema = createInsertSchema(processingQueue).omit({
  id: true,
  progress: true,
  startTime: true,
  endTime: true,
  estimatedCompletionTime: true,
  error: true,
});

export const insertMetadataQualitySchema = createInsertSchema(metadataQuality).omit({
  id: true,
});

// Types
export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

export type InsertDataset = z.infer<typeof insertDatasetSchema>;
export type Dataset = typeof datasets.$inferSelect;

export type InsertProcessingQueue = z.infer<typeof insertProcessingQueueSchema>;
export type ProcessingQueue = typeof processingQueue.$inferSelect;

export type InsertMetadataQuality = z.infer<typeof insertMetadataQualitySchema>;
export type MetadataQuality = typeof metadataQuality.$inferSelect;

// Additional custom types for the application
export type DatasetWithQuality = Dataset & {
  quality?: MetadataQuality;
  processing?: ProcessingQueue;
};

export type StatsData = {
  totalDatasets: number;
  processedDatasets: number;
  fairCompliantDatasets: number;
  queuedDatasets: number;
};
