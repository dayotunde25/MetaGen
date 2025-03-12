import { pgTable, text, serial, integer, boolean, timestamp, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// User schema for authentication
export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
  email: text("email").notNull(),
  fullName: text("full_name"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
  email: true,
  fullName: true,
});

// Dataset schema
export const datasets = pgTable("datasets", {
  id: serial("id").primaryKey(),
  title: text("title").notNull(),
  description: text("description"),
  source: text("source").notNull(),
  sourceUrl: text("source_url").notNull(),
  category: text("category"),
  size: text("size"),
  formats: text("formats").array(),
  status: text("status").notNull().default("pending"), // pending, downloading, structuring, generating, processed, failed
  progress: integer("progress").default(0),
  estimatedTimeToCompletion: text("estimated_time_to_completion"),
  keywords: text("keywords").array(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const insertDatasetSchema = createInsertSchema(datasets).pick({
  title: true,
  description: true,
  source: true,
  sourceUrl: true,
  category: true,
  size: true,
  formats: true,
  keywords: true,
});

// DatasetMetadata schema
export const metadata = pgTable("metadata", {
  id: serial("id").primaryKey(),
  datasetId: integer("dataset_id").notNull(),
  schemaOrgJson: jsonb("schema_org_json").notNull(),
  fairAssessment: jsonb("fair_assessment"),
  dataStructure: jsonb("data_structure"),
  isFairCompliant: boolean("is_fair_compliant").default(false),
  creator: text("creator"),
  publisher: text("publisher"),
  publicationDate: text("publication_date"),
  lastUpdated: text("last_updated"),
  language: text("language"),
  license: text("license"),
  temporalCoverage: text("temporal_coverage"),
  spatialCoverage: text("spatial_coverage"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertMetadataSchema = createInsertSchema(metadata).pick({
  datasetId: true,
  schemaOrgJson: true,
  fairAssessment: true,
  dataStructure: true,
  isFairCompliant: true,
  creator: true,
  publisher: true,
  publicationDate: true,
  lastUpdated: true,
  language: true,
  license: true,
  temporalCoverage: true,
  spatialCoverage: true,
});

// Processing History schema
export const processingHistory = pgTable("processing_history", {
  id: serial("id").primaryKey(),
  datasetId: integer("dataset_id").notNull(),
  operation: text("operation").notNull(), // download, structure, metadata_generation
  status: text("status").notNull(), // success, failed, in_progress
  details: text("details"),
  startTime: timestamp("start_time").defaultNow().notNull(),
  endTime: timestamp("end_time"),
});

export const insertProcessingHistorySchema = createInsertSchema(processingHistory).pick({
  datasetId: true,
  operation: true,
  status: true,
  details: true,
  startTime: true,
  endTime: true,
});

export type User = typeof users.$inferSelect;
export type InsertUser = z.infer<typeof insertUserSchema>;

export type Dataset = typeof datasets.$inferSelect;
export type InsertDataset = z.infer<typeof insertDatasetSchema>;

export type Metadata = typeof metadata.$inferSelect;
export type InsertMetadata = z.infer<typeof insertMetadataSchema>;

export type ProcessingHistory = typeof processingHistory.$inferSelect;
export type InsertProcessingHistory = z.infer<typeof insertProcessingHistorySchema>;
