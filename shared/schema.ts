import { pgTable, text, serial, integer, boolean, timestamp, json, varchar, real } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// User schema
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
  name: text("name").notNull(),
  description: text("description").notNull(),
  source: text("source").notNull(),
  sourceUrl: text("source_url").notNull(),
  format: text("format").notNull(),
  size: text("size").notNull(),
  records: text("records"),
  lastUpdated: text("last_updated").notNull(),
  fairScore: integer("fair_score").notNull(),
  tags: text("tags").array().notNull(),
  schema: text("schema").notNull(),
  license: text("license").notNull(),
  citation: text("citation"),
  quality: text("quality").notNull(),
  thumbnail: text("thumbnail"),
  status: text("status").notNull().default('available'),
  createdAt: timestamp("created_at").notNull().defaultNow(),
  metadata: json("metadata")
});

export const insertDatasetSchema = createInsertSchema(datasets).omit({
  id: true,
  createdAt: true
});

export type InsertDataset = z.infer<typeof insertDatasetSchema>;
export type Dataset = typeof datasets.$inferSelect;

// Processing Queue schema
export const processingQueue = pgTable("processing_queue", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  source: text("source").notNull(),
  sourceUrl: text("source_url").notNull(),
  status: text("status").notNull().default('queued'),
  progress: integer("progress").notNull().default(0),
  size: text("size"),
  createdAt: timestamp("created_at").notNull().defaultNow(),
  estimatedCompletionTime: text("estimated_completion_time"),
  error: text("error")
});

export const insertProcessingQueueSchema = createInsertSchema(processingQueue).omit({
  id: true,
  progress: true,
  createdAt: true,
  error: true
});

export type InsertProcessingQueue = z.infer<typeof insertProcessingQueueSchema>;
export type ProcessingQueue = typeof processingQueue.$inferSelect;

// Search schema
export const searchHistory = pgTable("search_history", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id),
  query: text("query").notNull(),
  filters: json("filters"),
  createdAt: timestamp("created_at").notNull().defaultNow()
});

export const insertSearchHistorySchema = createInsertSchema(searchHistory).omit({
  id: true,
  createdAt: true
});

export type InsertSearchHistory = z.infer<typeof insertSearchHistorySchema>;
export type SearchHistory = typeof searchHistory.$inferSelect;

// Repository schema
export const repositories = pgTable("repositories", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  url: text("url").notNull(),
  apiEndpoint: text("api_endpoint"),
  description: text("description"),
  icon: text("icon")
});

export const insertRepositorySchema = createInsertSchema(repositories).omit({
  id: true
});

export type InsertRepository = z.infer<typeof insertRepositorySchema>;
export type Repository = typeof repositories.$inferSelect;
