import { pgTable, text, serial, integer, boolean, jsonb, timestamp } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Existing user schema
export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

// Dataset schema
export const datasets = pgTable("datasets", {
  id: serial("id").primaryKey(),
  title: text("title").notNull(),
  description: text("description").notNull(),
  source: text("source").notNull(),
  sourceUrl: text("source_url").notNull(),
  dataType: text("data_type").notNull(),
  category: text("category").notNull(),
  size: text("size"),
  format: text("format"),
  recordCount: integer("record_count"),
  fairCompliant: boolean("fair_compliant").default(false),
  metadataQuality: integer("metadata_quality").default(0),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertDatasetSchema = createInsertSchema(datasets).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDataset = z.infer<typeof insertDatasetSchema>;
export type Dataset = typeof datasets.$inferSelect;

// Metadata schema
export const metadata = pgTable("metadata", {
  id: serial("id").primaryKey(),
  datasetId: integer("dataset_id").notNull(),
  schemaOrgJson: jsonb("schema_org_json").notNull(),
  fairScores: jsonb("fair_scores").notNull(),
  keywords: text("keywords").array(),
  variableMeasured: text("variable_measured").array(),
  temporalCoverage: text("temporal_coverage"),
  spatialCoverage: jsonb("spatial_coverage"),
  license: text("license"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertMetadataSchema = createInsertSchema(metadata).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertMetadata = z.infer<typeof insertMetadataSchema>;
export type Metadata = typeof metadata.$inferSelect;

// SearchQuery schema for NLP search
export const searchQueries = pgTable("search_queries", {
  id: serial("id").primaryKey(),
  query: text("query").notNull(),
  processedQuery: text("processed_query").notNull(),
  results: jsonb("results").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertSearchQuerySchema = createInsertSchema(searchQueries).omit({
  id: true,
  createdAt: true,
});

export type InsertSearchQuery = z.infer<typeof insertSearchQuerySchema>;
export type SearchQuery = typeof searchQueries.$inferSelect;
